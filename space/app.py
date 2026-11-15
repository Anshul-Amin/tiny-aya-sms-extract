"""Demo Space: Indian bank/UPI SMS -> JSON with fine-tuned Tiny Aya (runs on free CPU).

Backends, in order: GGUF via llama-cpp-python (fast on CPU) -> transformers + LoRA (slow fallback).
Shows the model's JSON, schema validation, and the regex baseline side by side.
"""
import json
import os
import time

import gradio as gr

from regex_baseline import predict as regex_predict
from scoring import extract_json, schema_errors
from smsgen import SYSTEM_PROMPT, build_messages

GGUF_REPO = os.environ.get("GGUF_REPO", "")
GGUF_FILE = os.environ.get("GGUF_FILE", "")
LORA_REPO = os.environ.get("LORA_REPO", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "CohereLabs/tiny-aya-global")

BACKEND, LLM, TOK = None, None, None


def load():
    global BACKEND, LLM, TOK
    if BACKEND:
        return
    if GGUF_REPO and GGUF_FILE:
        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            path = hf_hub_download(GGUF_REPO, GGUF_FILE)
            LLM = Llama(model_path=path, n_ctx=1024, n_threads=os.cpu_count() or 2, verbose=False)
            BACKEND = "gguf (llama.cpp, CPU)"
            return
        except Exception as e:
            print("GGUF load failed:", e)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    LLM = PeftModel.from_pretrained(base, LORA_REPO).eval()
    BACKEND = "transformers + LoRA (CPU, slow)"


def run_model(sms: str) -> str:
    load()
    msgs = build_messages(sms)
    if BACKEND.startswith("gguf"):
        out = LLM.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=200)
        return out["choices"][0]["message"]["content"]
    import torch
    prompt = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = TOK(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        gen = LLM.generate(**enc, max_new_tokens=200, do_sample=False)
    return TOK.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def extract(sms: str):
    if not sms or not sms.strip():
        return "{}", "Paste an SMS first.", "{}"
    t0 = time.time()
    raw = run_model(sms.strip())
    dt = time.time() - t0
    obj = extract_json(raw)
    errs = schema_errors(obj)
    status = (f"✅ valid JSON, schema OK · {dt:.1f}s · {BACKEND}" if obj is not None and not errs
              else f"⚠️ {', '.join(errs)} · {dt:.1f}s · {BACKEND}")
    pretty = json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else raw
    rx = json.dumps(json.loads(regex_predict(sms)), ensure_ascii=False, indent=2)
    return pretty, status, rx


EXAMPLES = [
    "Rs.450.00 debited from A/c XX1234 on 14-11-26 to SWIGGY via UPI. UPI Ref 401234567890. Avl bal Rs.12,340.50",
    "आपके खाते XX७७१२ में ₹२५,०००.०० NEFT द्वारा VEDA HOSPITALS से जमा हुए, दिनांक ०१-११-२०२६। शेष ₹३१,४०२.१०।",
    "Payment fail! Rs 2,100 OLA ko nahi gaya. Paisa kata ho to 3 din mein wapas. UPI Ref 990011223344",
    "Kestrel Bank CC xx4410: Rs.2,999 spent at CROMA on 11-11-26. Avl Lmt Rs.47,001. Total Lmt Rs.50,000.",
    "Do not share. 482913 is OTP for txn of INR 12,500.00 at AMAZON on card XX4410. -Kestrel Bank",
    "Card XX5521 se USD 12.99 ka payment GOOGLE ONE ko hua, 01-11-2026.",
    "Kal 15-11-26 ko aapke A/c XX0702 se Rs 2,300 ki EMI kategi. Balance maintain rakhein.",
]

with gr.Blocks(title="Tiny Aya · SMS → JSON") as demo:
    gr.Markdown(
        "# Tiny Aya · Indian bank & UPI SMS → JSON\n"
        "Fine-tuned **Tiny Aya Global (3.35B)** with Lexsi **CuratorKIT → AlignTune → AuditKIT**. "
        "Works in **English, Hindi and Hinglish**, and returns `is_transaction: false` for OTPs, offers, "
        "reminders and future debits. Runs on a free CPU. All demo data is synthetic, and bank names are fictional.\n\n"
        "⚠️ *Please don't paste real messages that contain personal information.*")
    with gr.Row():
        sms = gr.Textbox(label="SMS", lines=4, placeholder="Paste a bank / UPI SMS…")
    btn = gr.Button("Extract", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        out = gr.Code(label="Fine-tuned Tiny Aya", language="json")
        rx = gr.Code(label="Regex baseline (for comparison)", language="json")
    gr.Examples(EXAMPLES, inputs=sms)
    with gr.Accordion("System prompt & schema", open=False):
        gr.Markdown(f"```\n{SYSTEM_PROMPT}\n```")
    btn.click(extract, inputs=sms, outputs=[out, status, rx])

if __name__ == "__main__":
    demo.launch()
