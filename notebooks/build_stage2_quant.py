"""Builds notebooks/stage2_quant.ipynb: CircuitKIT circuit discovery + circuit-aware mixed-precision quantization."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()


cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# Stage 2 · Smaller models for smaller GPUs: CircuitKIT circuit-aware quantization

**Tiny Aya SMS Extract** · Anshul J Amin · Lexsi AI Safety & Alignment Hackathon, Track 1

Bank SMS are private, so the extractor should run on small, local hardware. This notebook builds versions of the fine-tuned model for 16 / 8 / 4 / 2 GB GPUs and measures what each loses.

1. **Merge** the LoRA adapter into Tiny Aya Fire (fp16).
2. **CircuitKIT** finds the circuit behind the model's hardest decision, *debit vs credit*: contrastive pairs are the same SMS with "debited" ↔ "credited", scored at the `"type"` token. Node-level EAP-IG gives an importance score per attention head and MLP; we sum them per layer.
3. **Quantize** with and without protection: the most important layers stay at higher precision, the rest are compressed. A *random-protection* control (same number of layers, chosen at random) shows whether the circuit is what matters.
4. **Evaluate** every version with the same scorer on unseen formats and the hand-written set, and record memory and speed.

GPU versions use bitsandbytes (fp16, int8, nf4); the 2 GB tier uses GGUF (llama.cpp), whose quantization also compresses the large multilingual embedding table.

**How to run (Kaggle):** *File → Import Notebook* → this file. **GPU T4 x1**, **Internet ON**, secret `HF_TOKEN` (write). Run `QUICK_MODE = True` first, then `QUICK_MODE = False`.
""")

code(r'''
import os, sys, json, time, random, subprocess, gc, glob, shutil, traceback, re, math
from collections import Counter

QUICK_MODE = True
SEED = 7
BASE_MODEL   = "CohereLabs/tiny-aya-fire"
ADAPTER_REPO = "Meowniece/tiny-aya-sms-extract-fire-lora"      # switch to the -fire-v3-lora repo if v3 wins (decision rule)
DATA_REPO    = "Meowniece/tiny-aya-sms-extract-v2-data"   # Stage 1 splits (identical test sets in v3)
TRAIN_FILE   = "train"
QUANT_REPO   = "Meowniece/tiny-aya-sms-extract-quantized"
MODEL_COLLECTION = "Meowniece/tiny-aya-sms-extraction-models-6af9a1a0ced10d8fb22b4ed6"
PUSH_TO_HUB  = True

N_PAIRS     = 64      # contrastive pairs for circuit discovery
N_PROTECT   = 4       # layers kept at higher precision (of 36)
RUN_CIRCUITKIT = True
RUN_GGUF    = True    # llama.cpp build + GGUF tiers (CPU evaluation)
GGUF_EVAL_N = 44      # SMS per GGUF version (CPU is slow); the hand-written set by default
EVAL_SPLITS = ["test_ood", "challenge"]
GEN_BATCH, MAX_NEW = 16, 200

if QUICK_MODE:
    N_PAIRS, GGUF_EVAL_N = 8, 4
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work_quant")
for d in ["src", "data", "results", "models", "figures"]:
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
random.seed(SEED)
print("work dir:", WORK, "| quick mode:", QUICK_MODE)
''')

md("## 1. Install")
code(r'''
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + " ".join(args))
    if r.returncode: print(r.stderr[-1500:])
    return r.returncode == 0

def secret(name):
    if os.environ.get(name): return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None
if secret("HF_TOKEN"): os.environ["HF_TOKEN"] = secret("HF_TOKEN")
print("HF_TOKEN", "found" if os.environ.get("HF_TOKEN") else "MISSING")
# CircuitKIT needs torch>=2.13 and transformers 5, which would break this kernel's preinstalled stack
# (torchvision mismatch). So it gets its own virtual environment and runs in a subprocess.
pip("uv")
CK_PY = f"{WORK}/ckenv/bin/python"
def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + " ".join(cmd[-3:]))
    if r.returncode: print(r.stderr[-1500:])
    return r.returncode == 0
OK_CK = _run([sys.executable, "-m", "uv", "venv", f"{WORK}/ckenv", "--python", sys.executable]) and \
        _run([sys.executable, "-m", "uv", "pip", "install", "--python", CK_PY, "circuitkit"])
if OK_CK:
    r = subprocess.run([CK_PY, "-c", "import torch, circuitkit; print('circuitkit', circuitkit.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"],
                       capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-600:])
    OK_CK = r.returncode == 0
pip("bitsandbytes", "peft", "accelerate", "huggingface_hub", "matplotlib", "sentencepiece", "protobuf", "requests")
# check versions in a separate process so this kernel imports transformers only once
r = subprocess.run([sys.executable, "-c", "import transformers, torch, peft; print(transformers.__version__, torch.__version__, peft.__version__)"],
                   capture_output=True, text=True)
print("transformers / torch / peft:", (r.stdout or r.stderr).strip()[-300:])
''')

md("## 2. Project source files")
for path in ["src/smsgen.py", "src/scoring.py"]:
    content = read(path)
    code(f"# ---- {path} ----\nCONTENT = {content!r}\n"
         f"open(f\"{{WORK}}/{path}\", \"w\", encoding=\"utf-8\").write(CONTENT)\n"
         f"print('wrote {path}', len(CONTENT), 'chars')")
code(r'''
sys.path.insert(0, f"{WORK}/src")
import importlib, smsgen, scoring as S
from smsgen import build_messages, gold_to_str, SYSTEM_PROMPT
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download, HfApi, login
if os.environ.get("HF_TOKEN"): login(token=os.environ["HF_TOKEN"])
api = HfApi()
def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
splits = {}
for name in [TRAIN_FILE] + EVAL_SPLITS:
    try:
        splits[name] = load_jsonl(hf_hub_download(DATA_REPO, f"data/{name}.jsonl", repo_type="dataset"))
    except Exception as e:
        print(name, "not available:", str(e)[:80])
EVAL_SPLITS = [s for s in EVAL_SPLITS if splits.get(s)]
if QUICK_MODE:
    splits = {k: (v if k == TRAIN_FILE else v[:6]) for k, v in splits.items()}
print({k: len(v) for k, v in splits.items()}, "| eval:", EVAL_SPLITS)
''')

md("## 3. Merge the adapter into the base model (fp16)")
code(r'''
from peft import PeftModel
MERGED = f"{WORK}/models/merged_fp16"
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
if not os.path.exists(f"{MERGED}/config.json"):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(base, ADAPTER_REPO).merge_and_unload()
    merged.save_pretrained(MERGED, safe_serialization=True); tok.save_pretrained(MERGED)
    del merged, base; gc.collect()
N_LAYERS = json.load(open(f"{MERGED}/config.json"))["num_hidden_layers"]
size_gb = sum(os.path.getsize(f) for f in glob.glob(f"{MERGED}/*.safetensors")) / 1e9
print(f"merged model: {N_LAYERS} layers, {size_gb:.2f} GB on disk")
''')

md("## 4. Contrastive pairs: debit ↔ credit\nThe clean prompt is the exact chat prompt the model was trained on, followed by the start of its answer up to the `type` value. The corrupted prompt is the same SMS with the debit word swapped for its credit counterpart. Only pairs with identical token length and single-token answers are kept, as EAP requires.")
code(r'''
SWAPS = [("debited", "credited"), ("Debited", "Credited"), ("DEBITED", "CREDITED"), ("डेबिट", "क्रेडिट"),
         ("debit hue", "credit hue"), ("debit hua", "credit hua")]
PREFIX = '{"is_transaction": true, "type": "'
def chat_prompt(sms):
    return tok.apply_chat_template(build_messages(sms), tokenize=False, add_generation_prompt=True) + PREFIX
first = lambda s: tok(s, add_special_tokens=False)["input_ids"]
d_ids, c_ids = first("debit"), first("credit")
ANS_CLEAN, ANS_CORR = tok.decode(d_ids[:1]), tok.decode(c_ids[:1])
print("answer tokens:", d_ids, c_ids, "->", repr(ANS_CLEAN), repr(ANS_CORR))
pairs, rng = [], random.Random(SEED)
cands = [r for r in splits[TRAIN_FILE] if r["gold"]["is_transaction"] and r["gold"]["type"] == "debit"]
rng.shuffle(cands)
for r in cands:
    for a, b in SWAPS:
        if a in r["sms"]:
            cp, xp = chat_prompt(r["sms"]), chat_prompt(r["sms"].replace(a, b))
            if len(first(cp)) == len(first(xp)):
                pairs.append({"clean_prompt": cp, "clean_answer": ANS_CLEAN, "corrupt_prompt": xp, "corrupt_answer": ANS_CORR,
                              "id": r["id"], "lang": r["lang"]})
            break
    if len(pairs) >= N_PAIRS: break
write_jsonl(f"{WORK}/data/circuit_pairs.jsonl", pairs)
print(len(pairs), "aligned pairs | by language:", dict(Counter(p["lang"] for p in pairs)))
print("example corrupted SMS tail:", pairs[0]["corrupt_prompt"][-260:] if pairs else None)
''')

md("## 5. CircuitKIT discovery (node-level EAP-IG, run in its own process)\nFloat16 on a T4 (bfloat16 is not native there), batch size 1, 3 IG steps, CircuitKIT's TransformerLens memory patch on. The result is one importance score per attention head and MLP block, summed per layer.")
code(r'''
DISCOVER = r"""
import os, sys, json
os.environ["CIRCUITKIT_TL_MEMORY_PATCH"] = "1"
import circuitkit as ck
from circuitkit import register_task
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, DatasetShape
W, merged = sys.argv[1], sys.argv[2]
pairs = [json.loads(l) for l in open(f"{W}/data/circuit_pairs.jsonl", encoding="utf-8")]
recs = [ContrastiveRecord(record_id=str(i), clean_prompt=p["clean_prompt"], clean_answer=p["clean_answer"],
                          corrupt_prompt=p["corrupt_prompt"], corrupt_answer=p["corrupt_answer"]) for i, p in enumerate(pairs)]
ds = NormalizedDataset(name="sms_direction", shape=DatasetShape.QA, records=recs)
register_task(NormalizedTaskSpec(ds, name="sms_direction", templated=True))
model = ck.load_model(merged, dtype="float16")
circuit = ck.discover(model, "sms_direction", algorithm="eap-ig", level="node", n_examples=len(pairs),
                      batch_size=1, output_path=f"{W}/results/circuit_sms_direction.pt")
json.dump({"scores": circuit.scores, "top": circuit.top_nodes(20)}, open(f"{W}/results/circuit_scores.json", "w"), indent=1)
print("top nodes:", circuit.top_nodes(10))
"""
open(f"{WORK}/discover.py", "w").write(DISCOVER)
LAYER_SCORES, PROTECT, CK_OK = None, [], False
if RUN_CIRCUITKIT and OK_CK and pairs:
    t0 = time.time()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.run([CK_PY, f"{WORK}/discover.py", WORK, MERGED], capture_output=True, text=True, env=env)
    print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
    if proc.returncode == 0 and os.path.exists(f"{WORK}/results/circuit_scores.json"):
        sc = json.load(open(f"{WORK}/results/circuit_scores.json"))["scores"]
        LAYER_SCORES = [0.0] * N_LAYERS
        for node, v in sc.items():
            m = re.match(r"A(\d+)\.\d+", node) or re.match(r"MLP (\d+)", node)
            if m and int(m.group(1)) < N_LAYERS:
                LAYER_SCORES[int(m.group(1))] += abs(float(v))
        PROTECT = sorted(sorted(range(N_LAYERS), key=lambda i: -LAYER_SCORES[i])[:N_PROTECT])
        CK_OK = True
        print(f"CircuitKIT done in {(time.time() - t0) / 60:.1f} min | protected layers (top {N_PROTECT}):", PROTECT)
    else:
        print("⚠️ CircuitKIT discovery failed; circuit-protected versions are skipped (uniform versions still run)")
RANDOM_PROTECT = sorted(random.Random(SEED).sample([i for i in range(N_LAYERS) if i not in PROTECT] or list(range(N_LAYERS)), N_PROTECT))
print("random-protection control:", RANDOM_PROTECT)
json.dump({"layer_scores": LAYER_SCORES, "protect": PROTECT, "random_protect": RANDOM_PROTECT, "n_pairs": len(pairs),
           "circuitkit_ok": CK_OK}, open(f"{WORK}/results/protection.json", "w"), indent=1)
''')

md("## 6. GPU versions (bitsandbytes) and their evaluation")
code(r'''
def generate(model, prompts, batch=GEN_BATCH, max_new=MAX_NEW):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out

RESULTS = {}
def evaluate(name, model, extra=None):
    torch.cuda.reset_peak_memory_stats()
    row = {"name": name, **(extra or {})}
    try: row["weights_gb"] = round(model.get_memory_footprint() / 1e9, 2)
    except Exception: pass
    t0, n = time.time(), 0
    for sp in EVAL_SPLITS:
        rows = splits[sp]
        outs = generate(model, [tok.apply_chat_template(build_messages(r["sms"]), tokenize=False, add_generation_prompt=True) for r in rows])
        n += len(rows)
        sc = [S.score_example(o, r["gold"]) for o, r in zip(outs, rows)]
        m = S.aggregate(sc)
        ci = S.bootstrap_ci([float(x["exact_match"]) for x in sc], seed=SEED)
        row[sp] = {"em": m["exact_match"], "ci": ci, "halluc": m.get("hallucination_rate"), "schema": m["schema_valid"]}
        write_jsonl(f"{WORK}/results/predictions_{name}_{sp}.jsonl",
                    [{"id": r["id"], "output": o, "exact_match": s["exact_match"]} for r, o, s in zip(rows, outs, sc)])
    row["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    row["sms_per_s"] = round(n / (time.time() - t0), 2)
    RESULTS[name] = row
    print(name, "| weights", row.get("weights_gb"), "GB | peak", row["peak_gpu_gb"], "GB |",
          " ".join(f"{sp} EM {100 * row[sp]['em']:.1f}" for sp in EVAL_SPLITS), "|", row["sms_per_s"], "SMS/s")
    json.dump(RESULTS, open(f"{WORK}/results/quant_results.json", "w"), indent=1, default=str)

def skip_layers(layers):
    return ["lm_head"] + [f"model.layers.{i}" for i in layers]

VARIANTS = [("fp16", None, {"precision": "fp16"}),
            ("int8", BitsAndBytesConfig(load_in_8bit=True), {"precision": "int8 (LLM.int8)"}),
            ("nf4_uniform", BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), {"precision": "nf4, all layers"})]
if CK_OK:
    for tag, layers in [("nf4_circuit", PROTECT), ("nf4_random", RANDOM_PROTECT)]:
        VARIANTS.append((tag, BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                                 llm_int8_skip_modules=skip_layers(layers)),
                         {"precision": f"nf4, layers {layers} kept fp16"}))
for name, qcfg, extra in VARIANTS:
    try:
        kw = {"quantization_config": qcfg} if qcfg else {"torch_dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(MERGED, device_map={"": 0}, **kw).eval()
        evaluate(name, model, extra)
    except Exception:
        traceback.print_exc(); print(f"⚠️ {name} failed; continuing")
    finally:
        try: del model
        except NameError: pass
        gc.collect(); torch.cuda.empty_cache()
''')

md("## 7. GGUF versions for CPU and the 2 GB tier (llama.cpp)\nllama.cpp quantization also compresses the 262k-token embedding table (about 1 GB in fp16), which is what makes a 2 GB version possible. Circuit-protected versions keep every tensor of the protected layers at Q8_0 via `--tensor-type`. Evaluated on CPU (Kaggle's 4 vCPUs) with `llama-server`.")
code(r'''
GGUF = {}
if RUN_GGUF:
    try:
        LC = f"{WORK}/llama.cpp"
        if not os.path.isdir(LC):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LC], check=True)
        t0 = time.time()
        subprocess.run(["cmake", "-B", f"{LC}/build", "-S", LC, "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"],
                       check=True, capture_output=True)
        subprocess.run(["cmake", "--build", f"{LC}/build", "-j", "4", "--target", "llama-quantize", "llama-server"],
                       check=True, capture_output=True)
        print(f"llama.cpp built in {(time.time() - t0) / 60:.1f} min")
        F16 = f"{WORK}/models/sms-f16.gguf"
        r = subprocess.run([sys.executable, f"{LC}/convert_hf_to_gguf.py", MERGED, "--outfile", F16, "--outtype", "f16"],
                           capture_output=True, text=True)
        print(r.stderr[-800:])
        def quant(name, qtype, protect=None):
            out = f"{WORK}/models/tiny-aya-sms-{name}.gguf"
            cmd = [f"{LC}/build/bin/llama-quantize"]
            if protect:
                cmd += ["--tensor-type", "blk\\.(" + "|".join(map(str, protect)) + ")\\..*=q8_0"]
            r = subprocess.run(cmd + [F16, out, qtype], capture_output=True, text=True)
            if r.returncode or not os.path.exists(out):
                print(name, "quantize failed:", r.stderr[-600:]); return
            GGUF[name] = {"path": out, "gb": round(os.path.getsize(out) / 1e9, 2), "type": qtype, "protect": protect or []}
            print(name, GGUF[name]["gb"], "GB")
        quant("q8_0", "Q8_0")
        quant("q4_k_m", "Q4_K_M")
        quant("q3_k_m", "Q3_K_M")
        if CK_OK:
            quant("q4_k_m-circuit", "Q4_K_M", PROTECT)
            quant("q3_k_m-circuit", "Q3_K_M", PROTECT)
            quant("q3_k_m-random", "Q3_K_M", RANDOM_PROTECT)
    except Exception:
        traceback.print_exc(); print("⚠️ GGUF step failed")
''')

code(r'''
import requests
def eval_gguf(name, info, rows):
    port = 8099
    srv = subprocess.Popen([f"{WORK}/llama.cpp/build/bin/llama-server", "-m", info["path"], "--port", str(port),
                            "-c", "2048", "-t", str(os.cpu_count() or 4), "--no-webui"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200: break
            except Exception: pass
            time.sleep(2)
        outs, t0 = [], time.time()
        for r in rows:
            prompt = tok.apply_chat_template(build_messages(r["sms"]), tokenize=False, add_generation_prompt=True)
            js = requests.post(f"http://127.0.0.1:{port}/completion", json={"prompt": prompt, "n_predict": MAX_NEW,
                               "temperature": 0, "cache_prompt": True}, timeout=600).json()
            outs.append(js.get("content", ""))
        sec = (time.time() - t0) / max(len(rows), 1)
    finally:
        srv.terminate(); srv.wait()
    sc = [S.score_example(o, r["gold"]) for o, r in zip(outs, rows)]
    m = S.aggregate(sc)
    RESULTS[f"gguf_{name}"] = {"name": f"gguf_{name}", "precision": info["type"] + (f", layers {info['protect']} kept Q8_0" if info["protect"] else ""),
                               "file_gb": info["gb"], "eval_rows": len(rows), "em": m["exact_match"],
                               "ci": S.bootstrap_ci([float(x["exact_match"]) for x in sc], seed=SEED),
                               "halluc": m.get("hallucination_rate"), "cpu_sec_per_sms": round(sec, 1)}
    print(f"gguf {name}: {info['gb']} GB | EM {100 * m['exact_match']:.1f} on {len(rows)} SMS | {sec:.1f} s/SMS on CPU")
    json.dump(RESULTS, open(f"{WORK}/results/quant_results.json", "w"), indent=1, default=str)

GGUF_ROWS = (splits.get("challenge") or splits[EVAL_SPLITS[0]])[:GGUF_EVAL_N]
for name, info in GGUF.items():
    try: eval_gguf(name, info, GGUF_ROWS)
    except Exception: traceback.print_exc()
''')

md("## 8. Results: size vs accuracy")
code(r'''
def pct(x): return "–" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.1f}"
lines = ["| Version | Precision | Size (GB) | Fits GPU | " + " | ".join(f"{sp} EM [95% CI]" for sp in EVAL_SPLITS) + " | SMS/s (T4) |",
         "|---|---|---|---|" + "---|" * len(EVAL_SPLITS) + "---|"]
def tier(gb):
    return "–" if gb is None else ("2 GB" if gb + 0.3 <= 2 else "4 GB" if gb + 0.4 <= 4 else "8 GB" if gb + 0.6 <= 8 else "16 GB")
for k, v in RESULTS.items():
    if k.startswith("gguf_"): continue
    gb = v.get("peak_gpu_gb")
    lines.append(f"| {k} | {v.get('precision')} | {v.get('weights_gb')} (peak {gb}) | {tier(gb)} | "
                 + " | ".join(f"{pct(v[sp]['em'])} [{pct(v[sp]['ci'][0])}–{pct(v[sp]['ci'][1])}]" for sp in EVAL_SPLITS) + f" | {v.get('sms_per_s')} |")
GPU_TABLE = "\n".join(lines)
gl = ["| GGUF version | Precision | File (GB) | Fits | EM on hand-written (n) [95% CI] | CPU s/SMS |", "|---|---|---|---|---|---|"]
for k, v in RESULTS.items():
    if not k.startswith("gguf_"): continue
    gl.append(f"| {k[5:]} | {v['precision']} | {v['file_gb']} | {tier(v['file_gb'])} | {pct(v['em'])} ({v['eval_rows']}) [{pct(v['ci'][0])}–{pct(v['ci'][1])}] | {v['cpu_sec_per_sms']} |")
GGUF_TABLE = "\n".join(gl)
print(GPU_TABLE + "\n\n" + GGUF_TABLE)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    sp0 = "challenge" if "challenge" in EVAL_SPLITS else EVAL_SPLITS[0]
    for k, v in RESULTS.items():
        x = v.get("file_gb") if k.startswith("gguf_") else v.get("weights_gb")
        y = v.get("em") if k.startswith("gguf_") else v[sp0]["em"]
        mk = "s" if "circuit" in k else ("x" if "random" in k else "o")
        ax.scatter(x, 100 * y, marker=mk); ax.annotate(k.replace("gguf_", ""), (x, 100 * y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("model size (GB)"); ax.set_ylabel(f"exact match, {sp0} (%)")
    ax.set_title("Size vs accuracy (square = circuit-protected, x = random-protected)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/size_vs_accuracy.png", dpi=160)
except Exception:
    traceback.print_exc()
open(f"{WORK}/results/quant_tables.md", "w").write("## GPU versions\n" + GPU_TABLE + "\n\n## GGUF versions\n" + GGUF_TABLE)
''')

md("## 9. Publish the quantized versions (4th model repo)")
code(r'''
prot = json.load(open(f"{WORK}/results/protection.json"))
CARD = f"""---
base_model: {BASE_MODEL}
license: cc-by-nc-4.0
language: [en, hi]
library_name: gguf
tags: [sms, upi, banking, information-extraction, structured-output, hinglish, quantized, gguf, circuitkit]
---
# Tiny Aya SMS Extract · quantized versions for 2 / 4 / 8 / 16 GB

Smaller versions of the fine-tuned Tiny Aya Fire SMS extractor (adapter: `{ADAPTER_REPO}`). Author: Anshul J Amin.

**Circuit-aware:** CircuitKIT (node-level EAP-IG on {prot['n_pairs']} debit↔credit contrastive pairs) ranked the layers by their role in the debit-vs-credit decision.
Protected layers (kept at higher precision): `{prot['protect']}`. Random-protection control: `{prot['random_protect']}`.

## GGUF files (llama.cpp, CPU or GPU)
{GGUF_TABLE}

## GPU versions (transformers + bitsandbytes, built at load time from the merged model)
{GPU_TABLE}

Load the circuit-protected 4-bit version:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         llm_int8_skip_modules=["lm_head"] + [f"model.layers.{{i}}" for i in {prot['protect']}])
# merge the adapter into the base first (see the GitHub README), then from_pretrained(merged_dir, quantization_config=cfg)
```
Prompt format: see `src/smsgen.py` (`build_messages`) in https://github.com/Anshul-Amin/tiny-aya-sms-extract.
Licence: CC-BY-NC 4.0 (inherits Tiny Aya). Data are synthetic; bank names are fictional.
"""
open(f"{WORK}/README_quant.md", "w").write(CARD)
if PUSH_TO_HUB and not QUICK_MODE:
    api.create_repo(QUANT_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=f"{WORK}/README_quant.md", path_in_repo="README.md", repo_id=QUANT_REPO)
    for name, info in GGUF.items():
        api.upload_file(path_or_fileobj=info["path"], path_in_repo=os.path.basename(info["path"]), repo_id=QUANT_REPO)
        print("uploaded", os.path.basename(info["path"]))
    api.upload_folder(folder_path=f"{WORK}/results", path_in_repo="results", repo_id=QUANT_REPO)
    api.upload_folder(folder_path=f"{WORK}/figures", path_in_repo="figures", repo_id=QUANT_REPO)
    try: api.add_collection_item(MODEL_COLLECTION, item_id=QUANT_REPO, item_type="model", exists_ok=True)
    except Exception as e: print("collection:", e)
    print("https://huggingface.co/" + QUANT_REPO)
else:
    print("QUICK_MODE: not uploaded")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "stage2_quant.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
