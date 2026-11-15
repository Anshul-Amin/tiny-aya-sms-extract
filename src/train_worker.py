"""Fine-tuning worker, run in a SEPARATE process so GPU memory is fully released afterwards
(a failed attempt can never leave a model stuck on the GPU for the next attempt).

usage: python train_worker.py config.json
config keys: mode ("aligntune" | "trl"), out_dir, result_path, data (dict format -> jsonl path),
             kwargs (AlignTune kwargs), trl (TRL/PEFT settings), base_model
Writes result JSON: {"ok": bool, "path", "format", "artifact", "log_history", "error"}.
"""
import glob
import json
import os
import sys
import time
import traceback


def jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def get_log_history(trainer):
    for obj in (trainer, getattr(trainer, "trainer", None), getattr(trainer, "_trainer", None)):
        st = getattr(obj, "state", None) if obj is not None else None
        if st is not None and getattr(st, "log_history", None):
            return jsonable(st.log_history)
    return None


def find_artifact(out_dir):
    for pat in ["**/adapter_config.json", "**/config.json"]:
        hits = sorted(glob.glob(f"{out_dir}/**/{os.path.basename(pat)}", recursive=True), key=os.path.getmtime)
        if hits:
            return os.path.dirname(hits[-1])
    return None


def save_trainer(trainer, final_dir):
    for obj in (trainer, getattr(trainer, "trainer", None)):
        m = getattr(obj, "model", None) if obj is not None else None
        if m is not None and hasattr(m, "save_pretrained"):
            m.save_pretrained(final_dir)
            return final_dir
    for meth in ("save_model", "save"):
        if hasattr(trainer, meth):
            try:
                getattr(trainer, meth)(final_dir)
                return final_dir
            except Exception:
                pass
    return None


def training_diag(trainer):
    """What the trainer actually saw: rows, tokens of example 0, BOS count, tail (must end with the JSON answer)."""
    try:
        t = getattr(trainer, "trainer", trainer)
        tk = getattr(trainer, "tokenizer", None) or getattr(t, "processing_class", None)
        ds = t.train_dataset
        ids = ds[0]["input_ids"]
        return {"n_rows": len(ds), "n_tok": len(ids), "bos_in_first4": sum(i == tk.bos_token_id for i in ids[:4]),
                "tail": tk.decode(ids[-90:]), "max_length": getattr(t.args, "max_length", None),
                "fp16": bool(t.args.fp16), "bf16": bool(t.args.bf16)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def run_aligntune(cfg):
    from aligntune.core.backend_factory import create_sft_trainer
    errors = []
    for fmt in ["messages"]:   # the other formats would not match the chat template used at inference
        path = cfg["data"][fmt]
        kw = dict(cfg["kwargs"], dataset_name=path)
        if fmt == "text":
            kw["column_mapping"] = {"text": "text"}
        try:
            print(f"[worker] AlignTune: format={fmt} kwargs={sorted(kw)}", flush=True)
            trainer = create_sft_trainer(**kw)
            trainer.train()
            diag = training_diag(trainer)
            print("[worker] diag:", json.dumps(diag, ensure_ascii=False)[:1500], flush=True)
            final = save_trainer(trainer, f"{cfg['out_dir']}/final")
            art = find_artifact(cfg["out_dir"]) or final
            return {"ok": True, "path": "aligntune", "format": fmt, "artifact": art, "diag": diag,
                    "args": {k: str(v) for k, v in kw.items()}, "log_history": get_log_history(trainer)}
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors.append(f"{fmt}: {msg[:500]}")
            print(f"[worker]   failed: {msg[:500]}", flush=True)
            if "out of memory" in msg.lower():
                break        # a memory problem, not a data-format problem: stop trying formats
    return {"ok": False, "path": "aligntune", "error": " | ".join(errors)}


def run_trl(cfg):
    import inspect
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    t = cfg["trl"]
    bf16 = t["bf16"]
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype, device_map={"": 0})
    ck = dict(output_dir=cfg["out_dir"], num_train_epochs=t["epochs"], per_device_train_batch_size=t["batch"],
              gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"], lr_scheduler_type="cosine",
              logging_steps=10, save_strategy="no", report_to="none", seed=t["seed"],
              fp16=not bf16, bf16=bf16, gradient_checkpointing=True)
    sig = inspect.signature(SFTConfig).parameters
    if "warmup_ratio" in sig:
        ck["warmup_ratio"] = 0.05
    else:                       # newer transformers: a fractional warmup_steps means a ratio
        ck["warmup_steps"] = 0.05
    ck["max_length" if "max_length" in sig else "max_seq_length"] = t["max_len"]
    ds = load_dataset("json", data_files=cfg["data"]["messages"], split="train")
    trainer = SFTTrainer(model=base, args=SFTConfig(**ck), train_dataset=ds, processing_class=tok,
                         peft_config=LoraConfig(r=t["lora_r"], lora_alpha=t["lora_alpha"],
                                                lora_dropout=t["lora_dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]   # same as AlignTune's LoRA,
                                                task_type="CAUSAL_LM"))
    trainer.train()
    trainer.model.save_pretrained(f"{cfg['out_dir']}/final")
    diag = training_diag(trainer)
    print("[worker] diag:", json.dumps(diag, ensure_ascii=False)[:1500], flush=True)
    return {"ok": True, "path": "trl_fallback", "format": "messages", "artifact": f"{cfg['out_dir']}/final",
            "diag": diag, "log_history": get_log_history(trainer)}


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    t0 = time.time()
    try:
        res = run_aligntune(cfg) if cfg["mode"] == "aligntune" else run_trl(cfg)
    except Exception as e:
        traceback.print_exc()
        res = {"ok": False, "path": cfg["mode"], "error": f"{type(e).__name__}: {str(e)[:800]}"}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(cfg["result_path"], "w"), indent=1, default=str)
    print("[worker] result:", json.dumps({k: v for k, v in res.items() if k != "log_history"})[:1500], flush=True)
