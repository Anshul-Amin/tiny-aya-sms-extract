"""Builds notebooks/sms_extraction_pipeline.ipynb with all source files embedded."""
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


# ---------------------------------------------------------------------------
md("""
# Tiny Aya → structured JSON from Indian bank & UPI SMS (English · Hindi · Hinglish)

**Lexsi AI Safety & Alignment Hackathon · Track 1 · Problem 5 (Structured output & extraction)**, also Problem 2 (replacing a closed-model API) and Problem 4 (low-resource / multilingual / code-mixed).

Pipeline (3 Lexsi toolkits chained):

```
smsgen (synthetic data, exact labels) ─► CuratorKIT (clean, dedup, quality gate, provenance)
      ─► AuditKIT (baselines: regex · zero-shot · few-shot) ─► AlignTune (LoRA SFT on Tiny Aya Global)
      ─► AuditKIT (after) ─► stats · error analysis · ROI ─► HF collections + CPU demo Space
```

What makes this evaluation rigorous:
1. **Held-out templates (`test_ood`)**: message formats never seen in training, so we measure generalisation, not memorisation.
2. **Hand-written challenge set**: 44 human-labelled, messy, realistic messages in all three languages.
3. **Hallucination rate**: fields the model fills in although the SMS does not contain them.
4. **Hard negatives**: OTPs, offers, bill reminders, future debits and money requests mention amounts but are *not* transactions.
5. **Ablation**: the same fine-tune *without* hard negatives or noisy data, to show what the robustness data buys.

**How to run:** Kaggle → *New Notebook* → *File → Import notebook* → this file. Settings: **GPU T4 x1**, **Internet ON**. Add a Kaggle secret `HF_TOKEN` (a Hugging Face *write* token) and accept the Tiny Aya Global licence on its model page. Then *Run all*. Set `QUICK_MODE = True` first for a ~10-minute smoke test.
""")

code(r'''
# ============================== CONFIG (edit me) ==============================
import os, sys, json, time, math, random, subprocess, inspect, gc, glob, shutil, traceback

HF_USERNAME = ""                      # leave empty -> taken from your HF token
PROJECT     = "tiny-aya-sms-extract"  # prefix for all HF repos
BASE_MODEL  = "CohereLabs/tiny-aya-global"
SEED        = 7

QUICK_MODE  = False   # True = tiny subsets, ~10 min end-to-end smoke test. Run this FIRST.
RUN_ABLATION = True   # 2nd fine-tune without hard negatives/noise (+~25 min)
MAKE_GGUF   = True    # export a CPU-friendly GGUF for the free demo Space
PUSH_TO_HUB = True    # create HF model/dataset repos, collections and the demo Space
SPACE_ADD_TOKEN = False  # True = store your HF token as a Space secret (only needed if GGUF export fails)

N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD = 2400, 200, 400, 400
MAX_NEW_TOKENS = 160
GEN_BATCH      = 32
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, EPOCHS, TRAIN_BATCH, GRAD_ACCUM, MAX_SEQ_LEN = 2e-4, 1, 4, 4, 512

# ROI assumptions (edit to your situation; sources in the article)
GPU_USD_PER_HOUR = 0.35            # typical on-demand price of a cloud T4
MONTHLY_SMS = 10_000_000           # realistic volume: a mid-size expense/lending app (~330k SMS/day)
API_PRICES = {                      # USD per 1M tokens (input, output), list prices Nov 2026
    "gpt-5-mini":   (0.25, 2.00),
    "gpt-5.4-mini": (0.75, 4.50),
}

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
os.makedirs(WORK, exist_ok=True)
for d in ["src", "data", "results", "figures", "curated", "models"]:
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
if QUICK_MODE:
    N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD = 160, 30, 40, 40
random.seed(SEED)
print("work dir:", WORK, "| quick mode:", QUICK_MODE)
''')

md("## 1. Install the Lexsi toolkits")
code(r'''
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + " ".join(args))
    if r.returncode:
        print(r.stderr[-2000:])
    return r.returncode == 0

# make the HF token visible to subprocesses (the Tiny Aya config is gated)
if not os.environ.get("HF_TOKEN"):
    for getter in ("kaggle", "colab"):
        try:
            if getter == "kaggle":
                from kaggle_secrets import UserSecretsClient
                os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
            else:
                from google.colab import userdata
                os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
            break
        except Exception:
            pass
print(sys.version, "| HF token found:", bool(os.environ.get("HF_TOKEN")))
if "transformers" in sys.modules:
    print("⚠️ transformers is already imported in this kernel. Run -> Restart kernel, then Run All again.")
OK_ALIGN   = pip("aligntune")
OK_CURATOR = pip("curatorkit")
OK_AUDIT   = pip("auditkit")
pip("peft", "trl", "accelerate", "datasets", "huggingface_hub", "matplotlib")

# Check Tiny Aya support in a SEPARATE process, so an upgrade never mixes two transformers
# versions inside this kernel (that causes "cannot import name ... from transformers.utils").
def _subprocess_check():
    code = ("import transformers, torch; from transformers import AutoConfig; "
            f"AutoConfig.from_pretrained('{BASE_MODEL}', token=__import__('os').environ.get('HF_TOKEN')); "
            "print(transformers.__version__, torch.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout or r.stderr).strip()[-600:]

ok, msg = _subprocess_check()
if not ok and "gated" not in msg.lower() and "401" not in msg:
    print("transformers can't read the Tiny Aya config -> upgrading (before importing it here):", msg[-200:])
    pip("-U", "transformers")
    ok, msg = _subprocess_check()
print("Tiny Aya config check:", "OK" if ok else "FAILED", "|", msg[-300:])

import torch, transformers   # first import in this kernel -> always the final installed version
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
''')

code(r'''
# Hugging Face login: Kaggle secret, Colab secret or env var HF_TOKEN
token = os.environ.get("HF_TOKEN")
try:
    from kaggle_secrets import UserSecretsClient
    token = token or UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
try:
    from google.colab import userdata
    token = token or userdata.get("HF_TOKEN")
except Exception:
    pass
from huggingface_hub import login, whoami, HfApi
if token:
    login(token=token)
    HF_USERNAME = HF_USERNAME or whoami()["name"]
    print("logged in as", HF_USERNAME)
else:
    print("No HF token found: model download (gated) and uploads will fail.")
api = HfApi()
''')

md("## 2. Project source files (generator, scoring, regex baseline, challenge set)\nThese are the same files as in the GitHub repo, embedded so the notebook runs on its own.")
for path in ["src/smsgen.py", "src/scoring.py", "src/regex_baseline.py"]:
    content = read(path)
    code(f"# ---- {path} ----\nCONTENT = {content!r}\n"
         f"open(f\"{{WORK}}/{path}\", \"w\", encoding=\"utf-8\").write(CONTENT)\n"
         f"print('wrote {path}', len(CONTENT), 'chars')")
code(f"CHALLENGE_JSONL = {read('data/challenge_set.jsonl')!r}\n"
     "open(f\"{WORK}/data/challenge_set.jsonl\", \"w\", encoding=\"utf-8\").write(CHALLENGE_JSONL)\n"
     "sys.path.insert(0, f\"{WORK}/src\")\n"
     "import importlib, smsgen, scoring as S, regex_baseline as RB\n"
     "for m in (smsgen, S, RB): importlib.reload(m)\n"
     "from smsgen import build_dataset, gold_to_str, build_messages, SYSTEM_PROMPT, FIELDS\n"
     "print('challenge rows:', CHALLENGE_JSONL.count(chr(10)))")

md("## 3. Generate the dataset (exact labels, template-level held-out split)")
code(r'''
from collections import Counter
ds = build_dataset(N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD, seed=SEED)
splits = {k: [e.to_json() for e in v] for k, v in ds.items()}
splits["challenge"] = [json.loads(l) for l in open(f"{WORK}/data/challenge_set.jsonl", encoding="utf-8")]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

for k, rows in splits.items():
    write_jsonl(f"{WORK}/data/{k}.jsonl", rows)

# leakage checks
train_sms = {r["sms"] for r in splits["train"]}
for k in ["dev", "test_id", "test_ood", "challenge"]:
    assert not train_sms & {r["sms"] for r in splits[k]}, f"leak into {k}"
assert not {r["template_id"] for r in splits["test_ood"]} & {r["template_id"] for r in splits["train"]}
print("no SMS overlap and no OOD-template overlap with train ✓")

summary = {k: {"n": len(v), "langs": dict(Counter(r["lang"] for r in v)),
               "non_txn": sum(not r["gold"]["is_transaction"] for r in v)} for k, v in splits.items()}
print(json.dumps(summary, indent=1, ensure_ascii=False))
print("edge-case tags in train:", Counter(t for r in splits["train"] for t in r["tags"]).most_common())
for r in random.Random(1).sample(splits["test_ood"], 3):
    print("\n", r["lang"], r["family"], r["tags"], "\n ", r["sms"], "\n ", gold_to_str(r["gold"]))
''')

md("## 4. CuratorKIT: clean, de-duplicate, quality-gate, provenance\nWe export the training split in Alpaca format, run CuratorKIT, then **gate** every curated row: its output must still be valid JSON identical to the gold label, or it is dropped.")
code(r'''
def to_alpaca(r):
    return {"instruction": SYSTEM_PROMPT, "input": f"SMS: {r['sms']}", "output": gold_to_str(r["gold"]), "id": r["id"]}

raw_path = f"{WORK}/data/train_alpaca_raw.jsonl"
write_jsonl(raw_path, [to_alpaca(r) for r in splits["train"]])
by_input = {f"SMS: {r['sms']}": r for r in splits["train"]}
by_id = {r["id"]: r for r in splits["train"]}

def make_config(cls, **kw):
    try:
        params = inspect.signature(cls).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        pass
    return cls(**kw)

def load_any(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    data = json.load(open(path, encoding="utf-8"))
    return data if isinstance(data, list) else data.get("data", [])

curation = {"tool": "CuratorKIT", "ran": False, "rows_in": len(splits["train"]),
            "rows_in_by_lang": dict(Counter(r["lang"] for r in splits["train"]))}

def find_exports(out_dir):
    return [f for f in glob.glob(f"{out_dir}/**/*", recursive=True)
            if os.path.isfile(f) and "alpaca" in f.lower() and f.endswith((".json", ".jsonl"))]

def run_curator(clean, dedup="exact"):
    """Run CuratorKIT; map exported rows back to our rows; apply our JSON quality gate."""
    from curatorkit import Curator, CuratorConfig
    out_dir = f"{WORK}/curated/{dedup}_clean{int(bool(clean))}"
    cfg = make_config(CuratorConfig, dataset=raw_path, dedup=dedup, clean=clean,
                      export_formats=["alpaca"], output_dir=out_dir)
    res = Curator(cfg).run()
    try: res.print_summary()
    except Exception: pass
    exported = [e for f in find_exports(out_dir) for e in load_any(f)]
    kept, dropped_gate = [], 0
    for e in exported:
        inp = e.get("input", "")
        src = by_id.get(e.get("id")) or by_input.get(inp) or by_input.get(f"SMS: {inp}")
        ok = src is not None and S.extract_json(e.get("output", "")) is not None \
             and S.score_example(e.get("output", ""), src["gold"])["exact_match"]
        if ok: kept.append(src)
        else: dropped_gate += 1
    # de-duplicate the mapping (exports may repeat a row) while keeping order
    seen, uniq = set(), []
    for r in kept:
        if r["id"] not in seen:
            seen.add(r["id"]); uniq.append(r)
    stats = {"clean": clean, "dedup": dedup, "rows_exported": len(exported), "rows_after_gate": len(uniq),
             "dropped_by_quality_gate": dropped_gate,
             "kept_by_lang": dict(Counter(r["lang"] for r in uniq)),
             "provenance_files": sorted({os.path.basename(p) for p in glob.glob(f"{out_dir}/**/*", recursive=True)
                                         if os.path.basename(p) in ("manifest.json", "lexsi_provenance.json",
                                                                    "checksums.txt", "dataset_card.md")})}
    return uniq, stats

def retention(stats):
    return {l: stats["kept_by_lang"].get(l, 0) / n for l, n in curation["rows_in_by_lang"].items()}

curated_rows = None
if OK_CURATOR:
    try:
        rows1, st1 = run_curator(clean=True)
        curation["runs"] = [st1]
        ret1 = retention(st1)
        print("retention by language with clean=True:", {k: round(v, 2) for k, v in ret1.items()})
        curated_rows, chosen = rows1, st1
        if min(ret1.values()) < 0.8:
            # CuratorKIT's default text cleaning is tuned for English and can drop Devanagari /
            # romanised Hindi rows. Re-run with clean=False (dedup + provenance only).
            print("⚠️ a language lost >20% of its rows with clean=True -> re-running CuratorKIT with clean=False")
            rows2, st2 = run_curator(clean=False)
            curation["runs"].append(st2)
            if len(rows2) > len(rows1):
                curated_rows, chosen = rows2, st2
        # Restore any remaining rows CuratorKIT dropped, provided they pass our own gate
        # (gold JSON valid and exact-duplicate-free); recorded so the report stays honest.
        kept_ids = {r["id"] for r in curated_rows}
        restored = [r for r in splits["train"] if r["id"] not in kept_ids
                    and S.score_example(gold_to_str(r["gold"]), r["gold"])["exact_match"]]
        if restored and min(retention({"kept_by_lang": dict(Counter(r["lang"] for r in curated_rows))}).values()) < 0.8:
            curated_rows = curated_rows + restored
            curation["restored_rows"] = len(restored)
            curation["restored_by_lang"] = dict(Counter(r["lang"] for r in restored))
        curation.update(ran=True, chosen_run=f"clean={chosen['clean']}", rows_final=len(curated_rows),
                        final_by_lang=dict(Counter(r["lang"] for r in curated_rows)))
        try:   # near-duplicate analysis (reported, not applied)
            _, st3 = run_curator(clean=False, dedup="minhash")
            curation["minhash_rows_kept"] = st3["rows_after_gate"]
        except Exception as e:
            curation["minhash_error"] = str(e)[:300]
    except Exception as e:
        traceback.print_exc()
        curation["error"] = str(e)[:500]
        curated_rows = None

if curated_rows is None:
    print("⚠️ CuratorKIT step failed; falling back to exact de-dup in pandas (recorded in the report).")
    seen, curated_rows = set(), []
    for r in splits["train"]:
        if r["sms"] not in seen:
            seen.add(r["sms"]); curated_rows.append(r)
    curation["fallback"] = "pandas exact dedup"
train_rows = curated_rows
write_jsonl(f"{WORK}/data/train_curated.jsonl", train_rows)
json.dump(curation, open(f"{WORK}/results/curation_report.json", "w"), indent=1)
print(json.dumps(curation, indent=1))
''')

md("## 5. Load Tiny Aya Global and define the systems")
code(r'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16 else torch.float16
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# One GPU holds the 3.35B model easily; splitting it across Kaggle's 2 GPUs only adds latency.
DEVICE_MAP = {"": 0} if torch.cuda.is_available() else "cpu"

def load_base(dtype=DTYPE):
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map=DEVICE_MAP)
    m.eval()
    return m

def to_prompt(messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:   # template without a system role: fold it into the first user turn
        sys_msg = messages[0]["content"]
        msgs = [dict(m) for m in messages[1:]]
        msgs[0]["content"] = sys_msg + "\n\n" + msgs[0]["content"]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def alpaca_prompt(sms):
    return f"### Instruction:\n{SYSTEM_PROMPT}\n\n### Input:\nSMS: {sms}\n\n### Response:\n"

from transformers import StoppingCriteria, StoppingCriteriaList

def json_closed(text):
    """True once the text contains one complete top-level JSON object."""
    start = text.find("{")
    if start < 0:
        return False
    depth, in_str, esc = 0, False, False
    for c in text[start:]:
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        elif c == '"': in_str = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return True
    return False

class StopAfterJson(StoppingCriteria):
    """Per-sequence stop as soon as a full JSON object has been generated (saves ~half the decode time)."""
    def __init__(self, prompt_len):
        self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        texts = tok.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return torch.tensor([json_closed(t) for t in texts], device=input_ids.device)

# Tiny Aya's config sets cache_implementation="hybrid" (a static cache). Recent transformers then
# torch.compile the forward pass and RECOMPILE for every new batch shape (~1 min per batch on a T4).
# A dynamic cache with compilation disabled is ~10x faster for our short, varied prompts.
GEN_EXTRA = None
_GEN_CANDIDATES = [dict(cache_implementation="dynamic", disable_compile=True),
                   dict(cache_implementation="dynamic"),
                   dict(disable_compile=True),
                   dict()]

def _generate_fast(mdl, enc, plen, max_new):
    global GEN_EXTRA
    base = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopAfterJson(plen)]))
    if GEN_EXTRA is not None:
        return mdl.generate(**enc, **base, **GEN_EXTRA)
    last = None
    for extra in _GEN_CANDIDATES:
        try:
            t0 = time.time()
            out = mdl.generate(**enc, **base, **extra)
            GEN_EXTRA = extra
            print(f"generation settings: {extra or 'model defaults'} (first batch {time.time() - t0:.1f}s)")
            return out
        except (TypeError, ValueError) as e:
            last = e
    raise last

GEN_TOKEN_BUDGET = 12000   # max (batch x padded prompt length) per batch: fits a 16 GB T4 with room for the KV cache

@torch.no_grad()
def _run_batch(prompts, mdl, max_new):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(mdl.device)
    plen = enc["input_ids"].shape[1]
    try:
        out = _generate_fast(mdl, enc, plen, max_new)
    except torch.cuda.OutOfMemoryError:
        del enc
        gc.collect(); torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        half = len(prompts) // 2           # split and retry instead of crashing the run
        return _run_batch(prompts[:half], mdl, max_new) + _run_batch(prompts[half:], mdl, max_new)
    return [tok.decode(out[j, plen:], skip_special_tokens=True) for j in range(len(prompts))]

def generate(prompts, mdl, batch=GEN_BATCH, max_new=MAX_NEW_TOKENS):
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lens[i])   # length bucketing
    res, cur = [None] * len(prompts), []
    def flush():
        if cur:
            for i, o in zip(cur, _run_batch([prompts[i] for i in cur], mdl, max_new)):
                res[i] = o
            cur.clear()
    for i in order:
        if cur and (len(cur) >= batch or (len(cur) + 1) * (lens[i] + max_new) > GEN_TOKEN_BUDGET):
            flush()
        cur.append(i)
    flush()
    return res

model = load_base()
probe = generate([to_prompt(build_messages("Rs.450.00 debited from A/c XX1234 on 14-11-26 to SWIGGY via UPI. Avl bal Rs.12,340.50"))], model)[0]
print("probe output:", probe[:300])
if not probe.strip():
    print("⚠️ empty output in", DTYPE, "-> reloading in float32 (fp16 overflow guard)")
    del model; gc.collect(); torch.cuda.empty_cache()
    DTYPE = torch.float32
    model = load_base(DTYPE)
''')

code(r'''
# Few-shot exemplars: fixed, drawn from TRAIN only (one per language, incl. one hard negative)
_rng = random.Random(SEED)
def _pick(lang, txn):
    pool = [r for r in train_rows if r["lang"] == lang and r["gold"]["is_transaction"] == txn] \
        or [r for r in splits["train"] if r["lang"] == lang and r["gold"]["is_transaction"] == txn] \
        or [r for r in splits["train"] if r["gold"]["is_transaction"] == txn]
    clean = [r for r in pool if set(r["tags"]) <= {"hard_negative"}]   # no noise in exemplars
    return _rng.choice(clean or pool)
FEW_SHOT = [(r["sms"], gold_to_str(r["gold"])) for r in [_pick("en", True), _pick("hi", True), _pick("hinglish", False)]]

CACHE = {}   # (system, sms) -> raw output
def run_system(name, sms_list, fn):
    todo = [s for s in dict.fromkeys(sms_list) if (name, s) not in CACHE]
    if todo:
        t0 = time.time()
        for s, o in zip(todo, fn(todo)):
            CACHE[(name, s)] = o
        print(f"  {name}: {len(todo)} generations in {time.time() - t0:.0f}s")
    return [CACHE[(name, s)] for s in sms_list]

SYSTEMS = {
    "regex":     lambda xs: [RB.predict(x) for x in xs],
    "zero_shot": lambda xs: generate([to_prompt(build_messages(x)) for x in xs], model),
    "few_shot":  lambda xs: generate([to_prompt(build_messages(x, FEW_SHOT)) for x in xs], model),
}
EVAL_SPLITS = ["test_id", "test_ood", "challenge"]
''')

md("## 6. AuditKIT evaluation harness\nAuditKIT runs every system on every split (its run record, fingerprint and headline are saved). Generations are batched and cached first, so AuditKIT's per-call model function just reads the cache. Our task-specific metrics are computed on the same predictions.")
code(r'''
ak = None
if OK_AUDIT:
    try:
        import auditkit as ak
    except Exception as e:
        print("AuditKIT import failed:", e)

def _jsonable(x):
    try:
        json.dumps(x); return x
    except Exception:
        return str(x)

AUDIT_RUNS = {}
def audit_and_score(system, split):
    rows = splits[split]
    sms = [r["sms"] for r in rows]
    outputs = run_system(system, sms, SYSTEMS[system])
    meta = {"auditkit": False}
    if ak is not None:
        try:
            samples = [ak.Sample(input=r["sms"], target=gold_to_str(r["gold"])) for r in rows]
            model_fn = lambda prompts: [CACHE.get((system, p)) or SYSTEMS[system]([p])[0] for p in prompts]
            result = ak.evaluate(samples, model=model_fn)
            AUDIT_RUNS[(system, split)] = result
            meta = {"auditkit": True, "headline": _jsonable(getattr(result, "headline", None)),
                    "fingerprint": _jsonable(getattr(result, "fingerprint", None))}
            try:
                meta["summary"] = str(result.summary())[:2000]
            except Exception:
                pass
        except Exception as e:
            meta["auditkit_error"] = str(e)[:300]
    scored = [S.score_example(o, r["gold"]) for o, r in zip(outputs, rows)]
    return {"metrics": S.aggregate(scored), "by_lang": S.breakdown(rows, scored, "lang"),
            "by_tag": S.breakdown(rows, scored, "tags"), "scored": scored, "outputs": outputs, "audit": meta}

RESULTS = {}
def evaluate_system(system, split_names=EVAL_SPLITS):
    for sp in split_names:
        RESULTS[(system, sp)] = audit_and_score(system, sp)
        m = RESULTS[(system, sp)]["metrics"]
        print(f"{system:>12} | {sp:<9} | EM {m['exact_match']:.3f} | field acc {m.get('field_acc_txn', float('nan')):.3f}"
              f" | halluc {m['hallucination_rate']:.3f} | non-txn F1 {m['nontxn_f1']:.3f} | schema {m['schema_valid']:.3f}")

for s in ["regex", "zero_shot", "few_shot"]:
    evaluate_system(s)
''')

md("## 7. Fine-tune with AlignTune (LoRA SFT)\nAlignTune's `create_sft_trainer` with the TRL backend. The cell inspects the function signature and passes only arguments it accepts. If AlignTune cannot run in this environment, it falls back to plain TRL + PEFT with the same hyper-parameters, and **the report records which path was used**.")
code(r'''
from datasets import Dataset

def build_train_dataset(rows):
    recs = []
    for r in rows:
        msgs = build_messages(r["sms"]) + [{"role": "assistant", "content": gold_to_str(r["gold"])}]
        prompt = to_prompt(build_messages(r["sms"]))
        recs.append({"messages": msgs, "text": prompt + gold_to_str(r["gold"]) + (tok.eos_token or ""),
                     "instruction": SYSTEM_PROMPT, "input": f"SMS: {r['sms']}", "output": gold_to_str(r["gold"])})
    return Dataset.from_list(recs)

def free_gpu():
    global model
    try: del model
    except NameError: pass
    gc.collect(); torch.cuda.empty_cache()

def find_artifact(out_dir):
    for pat in ["**/adapter_config.json", "**/config.json"]:
        hits = sorted(glob.glob(f"{out_dir}/{pat}", recursive=True), key=os.path.getmtime)
        if hits:
            return os.path.dirname(hits[-1])
    return None

def get_log_history(trainer):
    for obj in (trainer, getattr(trainer, "trainer", None), getattr(trainer, "_trainer", None)):
        st = getattr(obj, "state", None) if obj is not None else None
        if st is not None and getattr(st, "log_history", None):
            return _jsonable(st.log_history)
    return None

def train_aligntune(rows, out_dir):
    from aligntune.core.backend_factory import create_sft_trainer
    data_dir = f"{out_dir}_data"
    ds_full = build_train_dataset(rows)
    variants = {
        "messages": ds_full.select_columns(["messages"]),
        "text": ds_full.select_columns(["text"]),
        "alpaca": ds_full.select_columns(["instruction", "input", "output"]),
    }
    params = inspect.signature(create_sft_trainer).parameters
    explicit = {k for k, p in params.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    print("create_sft_trainer parameters:", sorted(explicit), "| **kwargs:", has_kwargs)
    core = dict(model_name=BASE_MODEL, backend="trl", num_epochs=EPOCHS, max_steps=-1,
                batch_size=TRAIN_BATCH, learning_rate=LR)
    extra = dict(output_dir=out_dir, max_seq_length=MAX_SEQ_LEN, gradient_accumulation_steps=GRAD_ACCUM,
                 use_peft=True, peft=True, use_lora=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                 lora_dropout=LORA_DROPOUT, gradient_checkpointing=True, seed=SEED,
                 fp16=not BF16, bf16=BF16, logging_steps=10, save_strategy="no", report_to="none")
    last_err = None
    for fmt, d in variants.items():
        path = f"{data_dir}/{fmt}"
        d.to_json(f"{path}.jsonl", force_ascii=False)
        for ds_ref in [f"{path}.jsonl", path]:
            if ds_ref == path:
                d.save_to_disk(path)
            kw = dict(core, dataset_name=ds_ref)
            if fmt == "text":
                extra_fmt = dict(dataset_text_field="text", text_field="text")
            else:
                extra_fmt = {}
            for k, v in {**extra, **extra_fmt}.items():
                if k in explicit:
                    kw[k] = v
            try:
                print(f"AlignTune attempt: format={fmt} dataset={ds_ref} args={sorted(kw)}")
                trainer = create_sft_trainer(**kw)
                trainer.train()
                saved = None
                for obj in (trainer, getattr(trainer, "trainer", None)):
                    m = getattr(obj, "model", None) if obj is not None else None
                    if m is not None and hasattr(m, "save_pretrained"):
                        m.save_pretrained(f"{out_dir}/final"); saved = f"{out_dir}/final"; break
                for meth in ("save_model", "save"):
                    if saved is None and hasattr(trainer, meth):
                        try:
                            getattr(trainer, meth)(f"{out_dir}/final"); saved = f"{out_dir}/final"
                        except Exception:
                            pass
                art = find_artifact(out_dir) or saved
                try:
                    ev = trainer.evaluate()
                except Exception:
                    ev = None
                return {"path": "aligntune", "format": fmt, "dataset_ref": ds_ref, "artifact": art,
                        "args": {k: str(v) for k, v in kw.items()}, "aligntune_eval": _jsonable(ev),
                        "log_history": get_log_history(trainer)}
            except Exception as e:
                last_err = e
                print("  failed:", str(e)[:400])
                gc.collect(); torch.cuda.empty_cache()
                if "out of memory" in str(e).lower():
                    raise RuntimeError("AlignTune ran out of GPU memory (likely full fine-tuning)") from e
    raise RuntimeError(f"AlignTune could not train: {last_err}")

def train_trl_fallback(rows, out_dir):
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE_MAP)
    cfg_kw = dict(output_dir=out_dir, num_train_epochs=EPOCHS, per_device_train_batch_size=TRAIN_BATCH,
                  gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, lr_scheduler_type="cosine",
                  warmup_ratio=0.05, logging_steps=10, save_strategy="no", report_to="none", seed=SEED,
                  fp16=not BF16 and DTYPE != torch.float32, bf16=BF16, gradient_checkpointing=True)
    sig = inspect.signature(SFTConfig).parameters
    cfg_kw["max_length" if "max_length" in sig else "max_seq_length"] = MAX_SEQ_LEN
    trainer = SFTTrainer(model=base, args=SFTConfig(**cfg_kw),
                         train_dataset=build_train_dataset(rows).select_columns(["messages"]),
                         processing_class=tok,
                         peft_config=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    trainer.model.save_pretrained(f"{out_dir}/final")
    return {"path": "trl_fallback", "format": "messages", "artifact": f"{out_dir}/final",
            "log_history": get_log_history(trainer)}

def finetune(rows, tag):
    out_dir = f"{WORK}/models/{tag}"
    free_gpu()
    t0 = time.time()
    info = None
    if OK_ALIGN:
        try:
            info = train_aligntune(rows, out_dir)
        except Exception as e:
            traceback.print_exc()
            print("⚠️ AlignTune path failed -> TRL+PEFT fallback (recorded in report)")
    if info is None:
        info = train_trl_fallback(rows, out_dir)
    info["train_minutes"] = round((time.time() - t0) / 60, 1)
    info["n_train"] = len(rows)
    save_train_log(info.pop("log_history", None), tag)
    print(json.dumps(info, indent=1))
    return info

def save_train_log(log, tag):
    """Training logs -> results/ (uploaded to the HF dataset and model repos) + loss curve."""
    json.dump(log or [], open(f"{WORK}/results/train_log_{tag}.json", "w"), indent=1)
    pts = [(e.get("step"), e.get("loss")) for e in (log or []) if isinstance(e, dict) and e.get("loss") is not None]
    if pts:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([p[0] for p in pts], [p[1] for p in pts])
        ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.set_title(f"Training loss ({tag})")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{WORK}/figures/train_loss_{tag}.png", dpi=150); plt.close(fig)
    print(f"training log: {len(log or [])} entries -> results/train_log_{tag}.json")

def load_finetuned(info):
    from peft import PeftModel
    art = info["artifact"]
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE_MAP)
    if art and os.path.exists(os.path.join(art, "adapter_config.json")):
        m = PeftModel.from_pretrained(base, art)
    else:
        del base; gc.collect(); torch.cuda.empty_cache()
        m = AutoModelForCausalLM.from_pretrained(art, torch_dtype=DTYPE, device_map=DEVICE_MAP)
    m.eval()
    return m

FT_INFO = finetune(train_rows, "ft_main")
json.dump(FT_INFO, open(f"{WORK}/results/train_info_main.json", "w"), indent=1)
''')

md("## 8. Evaluate the fine-tuned model\nThe prompt format (chat template vs Alpaca) is chosen on the **dev** split only, because the trainer may have formatted examples either way. The test splits are never used for any choice.")
code(r'''
model = load_finetuned(FT_INFO)
dev = splits["dev"]
fmt_scores = {}
for fmt, builder in {"chat": lambda x: to_prompt(build_messages(x)), "alpaca": alpaca_prompt}.items():
    outs = generate([builder(r["sms"]) for r in dev], model)
    fmt_scores[fmt] = S.aggregate([S.score_example(o, r["gold"]) for o, r in zip(outs, dev)])["exact_match"]
FT_FORMAT = max(fmt_scores, key=fmt_scores.get)
print("dev exact match by prompt format:", fmt_scores, "-> using", FT_FORMAT)
_build = (lambda x: to_prompt(build_messages(x))) if FT_FORMAT == "chat" else alpaca_prompt
SYSTEMS["fine_tuned"] = lambda xs: generate([_build(x) for x in xs], model)
evaluate_system("fine_tuned")

# throughput for the ROI section (batched greedy decoding on this GPU)
sample = [r["sms"] for r in splits["test_id"][:min(128, len(splits["test_id"]))]]
torch.cuda.synchronize() if torch.cuda.is_available() else None
t0 = time.time(); _outs = generate([_build(x) for x in sample], model); dt = time.time() - t0
SMS_PER_SEC = len(sample) / dt
GEN_TOKENS_PER_SEC = sum(len(tok(o, add_special_tokens=False)["input_ids"]) for o in _outs) / dt
print(f"throughput: {SMS_PER_SEC:.2f} SMS/s, {GEN_TOKENS_PER_SEC:.0f} generated tokens/s "
      f"(batch {GEN_BATCH}, greedy) on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
''')

md("## 9. Ablation: the same fine-tune without the robustness data\nSame size and hyper-parameters, but **no hard negatives, no noise, no dropped fields**. This isolates what the edge-case data contributes.")
code(r'''
if RUN_ABLATION:
    eval_sms = {r["sms"] for k in ["dev", "test_id", "test_ood", "challenge"] for r in splits[k]}
    abl = build_dataset(len(train_rows), 0, 0, 0, seed=SEED + 1, neg_share=0.0, noise_p=0.0,
                        allow_drop=False, exclude_sms=eval_sms)["train"]
    abl_rows = [e.to_json() for e in abl]
    write_jsonl(f"{WORK}/data/train_ablation_clean.jsonl", abl_rows)
    free_gpu()
    ABL_INFO = finetune(abl_rows, "ft_ablation_clean")
    json.dump(ABL_INFO, open(f"{WORK}/results/train_info_ablation.json", "w"), indent=1)
    model_abl = load_finetuned(ABL_INFO)
    SYSTEMS["ft_no_robust_data"] = lambda xs: generate([_build(x) for x in xs], model_abl)
    evaluate_system("ft_no_robust_data")
    del model_abl; gc.collect(); torch.cuda.empty_cache()
''')

md("## 10. Results tables, significance tests, charts")
code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

systems = [s for s in ["regex", "zero_shot", "few_shot", "ft_no_robust_data", "fine_tuned"] if (s, "test_id") in RESULTS]
KEYS = [("exact_match", "Exact match"), ("field_acc_txn", "Field acc."), ("hallucination_rate", "Halluc. ↓"),
        ("omission_rate", "Omission ↓"), ("nontxn_f1", "Non-txn F1"), ("schema_valid", "Schema valid")]

def pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"

def table(split):
    lines = ["| System | " + " | ".join(n for _, n in KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for s in systems:
        m = RESULTS[(s, split)]["metrics"]
        lines.append(f"| {s} | " + " | ".join(pct(m.get(k)) for k, _ in KEYS) + " |")
    return "\n".join(lines)

def em_ci(s, split):
    v = [float(x["exact_match"]) for x in RESULTS[(s, split)]["scored"]]
    return S.bootstrap_ci(v, seed=SEED)

TABLES = {sp: table(sp) for sp in EVAL_SPLITS}
for sp in EVAL_SPLITS:
    print(f"\n### {sp} (n={len(splits[sp])})\n{TABLES[sp]}")

# Paired significance: fine-tuned vs each other system (exact match per example)
SIG = {}
for sp in EVAL_SPLITS:
    ft = [x["exact_match"] for x in RESULTS[("fine_tuned", sp)]["scored"]]
    for s in systems:
        if s == "fine_tuned":
            continue
        other = [x["exact_match"] for x in RESULTS[(s, sp)]["scored"]]
        SIG[(s, sp)] = S.mcnemar_exact(other, ft)
sig_lines = ["| Comparison | Split | FT wins | Other wins | McNemar p |", "|---|---|---|---|---|"]
for (s, sp), r in SIG.items():
    sig_lines.append(f"| fine_tuned vs {s} | {sp} | {r['b_only']} | {r['a_only']} | {r['p_value']:.2g} |")
SIG_TABLE = "\n".join(sig_lines)
print("\n" + SIG_TABLE)

# AuditKIT paired comparison (if the installed version exposes it)
AK_COMPARE = {}
if ak is not None:
    for sp in EVAL_SPLITS:
        a, b = AUDIT_RUNS.get(("few_shot", sp)), AUDIT_RUNS.get(("fine_tuned", sp))
        for fn_name in ("compare", "compare_runs"):
            if a is not None and b is not None and hasattr(ak, fn_name):
                try:
                    AK_COMPARE[sp] = str(getattr(ak, fn_name)(a, b).summary())[:1500]; break
                except Exception as e:
                    AK_COMPARE[sp] = f"error: {e}"[:300]
print("AuditKIT compare:", AK_COMPARE if AK_COMPARE else "not available in this AuditKIT version")

# Per-language & per-edge-case tables on the hardest splits
def lang_table(split):
    langs = ["en", "hi", "hinglish"]
    lines = ["| System | " + " | ".join(f"{l} EM" for l in langs) + " |", "|---|" + "---|" * len(langs)]
    for s in systems:
        bl = RESULTS[(s, split)]["by_lang"]
        lines.append(f"| {s} | " + " | ".join(pct(bl.get(l, {}).get("exact_match")) for l in langs) + " |")
    return "\n".join(lines)

def tag_table(split, top=14):
    tags = Counter(t for r in splits[split] for t in r["tags"])
    tags = [t for t, _ in tags.most_common(top)]
    shown = [s for s in ["regex", "few_shot", "ft_no_robust_data", "fine_tuned"] if s in systems]
    lines = ["| Edge case (n) | " + " | ".join(shown) + " |", "|---|" + "---|" * len(shown)]
    for t in tags:
        n = sum(t in r["tags"] for r in splits[split])
        lines.append(f"| {t} ({n}) | " + " | ".join(pct(RESULTS[(s, split)]["by_tag"].get(t, {}).get("exact_match")) for s in shown) + " |")
    return "\n".join(lines)

LANG_TABLES = {sp: lang_table(sp) for sp in EVAL_SPLITS}
TAG_TABLES = {sp: tag_table(sp) for sp in ["test_ood", "challenge"]}
print("\n" + LANG_TABLES["test_ood"] + "\n\n" + TAG_TABLES["challenge"])

# Charts
def bar(metric, title, fname):
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        vals = [100 * (RESULTS[(s, sp)]["metrics"].get(metric) or 0) for sp in EVAL_SPLITS]
        ax.bar([x + i * w for x in range(len(EVAL_SPLITS))], vals, w, label=s)
    ax.set_xticks([x + w * (len(systems) - 1) / 2 for x in range(len(EVAL_SPLITS))])
    ax.set_xticklabels(["seen formats", "unseen formats", "hand-written"])
    ax.set_ylabel("%"); ax.set_title(title); ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/{fname}", dpi=160); plt.close(fig)

bar("exact_match", "Exact match (all 11 fields correct)", "exact_match.png")
bar("hallucination_rate", "Hallucinated fields (lower is better)", "hallucination.png")
bar("nontxn_f1", "Catching non-transactions (OTP, offers, reminders): F1", "nontxn_f1.png")
print("charts saved to", f"{WORK}/figures")
''')

md("## 11. Error analysis\nEvery fine-tuned failure on the unseen-format and hand-written splits, with the exact fields that went wrong.")
code(r'''
import csv
err_rows = []
for sp in ["test_ood", "challenge"]:
    res = RESULTS[("fine_tuned", sp)]
    for r, sc, out in zip(splits[sp], res["scored"], res["outputs"]):
        if not sc["exact_match"]:
            pred = S.extract_json(out) or {}
            wrong = [k for k, ok in sc["field_correct"].items() if not ok]
            err_rows.append({"split": sp, "id": r["id"], "lang": r["lang"], "tags": ";".join(r["tags"]),
                             "wrong_fields": ";".join(wrong), "sms": r["sms"],
                             "gold": gold_to_str(r["gold"]), "pred": out.strip()[:600]})
with open(f"{WORK}/results/error_analysis_fine_tuned.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(err_rows[0].keys()) if err_rows else ["split"])
    w.writeheader(); w.writerows(err_rows)
WRONG_FIELD_COUNTS = Counter(k for e in err_rows for k in e["wrong_fields"].split(";") if k)
print(len(err_rows), "errors | most common wrong fields:", WRONG_FIELD_COUNTS.most_common(6))
for e in err_rows[:8]:
    print("\n", e["split"], e["lang"], e["tags"], "| wrong:", e["wrong_fields"], "\n ", e["sms"], "\n  gold:", e["gold"], "\n  pred:", e["pred"][:300])
''')

md("## 12. ROI: self-hosted Tiny Aya vs a closed-model API\nRealistic monthly volume × measured throughput (SMS/s and tokens/s) → GPU hours × hourly price, vs per-token API billing. Token counts are measured with each model's own prompt (zero-shot instructions for the API, same as our system prompt). Note: GPT-5-family models may also bill hidden reasoning tokens as output, so API figures are a **lower bound**.")
code(r'''
def n_tokens(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])

sample_rows = splits["test_id"][:200]
avg_in = sum(n_tokens(SYSTEM_PROMPT) + n_tokens("SMS: " + r["sms"]) for r in sample_rows) / len(sample_rows)
avg_out = sum(n_tokens(gold_to_str(r["gold"])) for r in sample_rows) / len(sample_rows)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
gpu_hours = MONTHLY_SMS / (SMS_PER_SEC * 3600)
train_cost = (FT_INFO.get("train_minutes") or 0) / 60 * GPU_USD_PER_HOUR
ROI = {"monthly_sms": MONTHLY_SMS, "avg_input_tokens": round(avg_in, 1), "avg_output_tokens": round(avg_out, 1),
       "self_hosted": {"gpu": gpu_name, "usd_per_gpu_hour": GPU_USD_PER_HOUR,
                       "sms_per_sec": round(SMS_PER_SEC, 2), "generated_tokens_per_sec": round(GEN_TOKENS_PER_SEC, 1),
                       "gpu_hours_per_month": round(gpu_hours, 1),
                       "usd_per_month_on_demand": round(gpu_hours * GPU_USD_PER_HOUR, 2),
                       "usd_per_month_always_on_1gpu": round(730 * GPU_USD_PER_HOUR, 2),
                       "one_time_training_usd": round(train_cost, 2),
                       "usd_per_1M_sms": round(GPU_USD_PER_HOUR / (SMS_PER_SEC * 3600) * 1e6, 2)},
       "api": {}}
for name, (pin, pout) in API_PRICES.items():
    per_sms = (avg_in * pin + avg_out * pout) / 1e6
    ROI["api"][name] = {"usd_per_1M_sms": round(per_sms * 1e6, 2), "usd_per_month": round(per_sms * MONTHLY_SMS, 2),
                        "input_tokens_per_month_M": round(avg_in * MONTHLY_SMS / 1e6, 1),
                        "output_tokens_per_month_M": round(avg_out * MONTHLY_SMS / 1e6, 1)}
json.dump(ROI, open(f"{WORK}/results/roi.json", "w"), indent=1)
print(json.dumps(ROI, indent=1))
sh = ROI["self_hosted"]
ROI_TABLE = (
    f"Scenario: **{MONTHLY_SMS:,} SMS / month** · avg {avg_in:.0f} input + {avg_out:.0f} output tokens per SMS\n\n"
    "| Option | Throughput | GPU hours / month | Cost / month (USD) | Cost / 1M SMS (USD) |\n|---|---|---|---|---|\n"
    f"| Tiny Aya fine-tuned, self-hosted, pay-per-hour ({sh['gpu']} @ ${GPU_USD_PER_HOUR}/h) | {sh['sms_per_sec']} SMS/s · {sh['generated_tokens_per_sec']:.0f} tok/s | {sh['gpu_hours_per_month']:,.1f} | {sh['usd_per_month_on_demand']:,.2f} | {sh['usd_per_1M_sms']:,.2f} |\n"
    f"| Tiny Aya fine-tuned, one always-on GPU (730 h) | same | 730 | {sh['usd_per_month_always_on_1gpu']:,.2f} | – |\n"
    + "\n".join(f"| {k} API (list price; excludes reasoning tokens) | – | – | {v['usd_per_month']:,.2f} | {v['usd_per_1M_sms']:,.2f} |" for k, v in ROI["api"].items())
    + f"\n\nOne-time fine-tuning cost: {FT_INFO.get('train_minutes')} GPU-minutes ≈ ${sh['one_time_training_usd']:.2f}."
)
print(ROI_TABLE)
''')

md("## 13. Save everything, then publish to the Hugging Face Hub\nModel repos (LoRA adapter, GGUF for CPU), dataset repo (all splits, curated train set, every prediction, metrics), two collections, and the demo Space.")
code(r'''
# ---- save metrics & predictions locally ----
all_metrics = {f"{s}/{sp}": {"metrics": v["metrics"], "by_lang": v["by_lang"], "audit": v["audit"]}
               for (s, sp), v in RESULTS.items()}
json.dump(all_metrics, open(f"{WORK}/results/metrics.json", "w"), indent=1, ensure_ascii=False, default=str)
for (s, sp), v in RESULTS.items():
    write_jsonl(f"{WORK}/results/predictions_{s}_{sp}.jsonl",
                [{"id": r["id"], "sms": r["sms"], "gold": r["gold"], "output": o, "exact_match": sc["exact_match"]}
                 for r, o, sc in zip(splits[sp], v["outputs"], v["scored"])])
json.dump({"roi": ROI, "significance": {f"{a}/{b}": v for (a, b), v in SIG.items()},
           "ft_prompt_format": FT_FORMAT, "dev_format_scores": fmt_scores, "train_info": FT_INFO,
           "auditkit_compare": AK_COMPARE},
          open(f"{WORK}/results/summary.json", "w"), indent=1, default=str)
print("saved results to", f"{WORK}/results")
''')

code(r'''
MODEL_REPO = f"{HF_USERNAME}/{PROJECT}-lora"
GGUF_REPO  = f"{HF_USERNAME}/{PROJECT}-gguf"
DATA_REPO  = f"{HF_USERNAME}/{PROJECT}-data"
SPACE_REPO = f"{HF_USERNAME}/{PROJECT}-demo"
ABL_REPO   = f"{HF_USERNAME}/{PROJECT}-lora-ablation-clean"

def headline(split, s="fine_tuned"):
    return RESULTS[(s, split)]["metrics"]

MODEL_CARD = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: cc-by-nc-4.0
language: [en, hi]
tags: [lexsi-hackathon, structured-output, information-extraction, upi, banking, hinglish, lora, aligntune]
---
# Tiny Aya Global, fine-tuned for Indian bank/UPI SMS → JSON

LoRA adapter for `{BASE_MODEL}` trained with **AlignTune** on data curated with **CuratorKIT** and evaluated with **AuditKIT**.
Input: one SMS in English, Hindi or Hinglish. Output: one JSON object with 11 fields (see the dataset card).

| Split | Exact match | Field accuracy | Hallucination ↓ | Non-txn F1 |
|---|---|---|---|---|
""" + "\n".join(f"| {sp} | {pct(headline(sp)['exact_match'])} | {pct(headline(sp).get('field_acc_txn'))} | {pct(headline(sp)['hallucination_rate'])} | {pct(headline(sp)['nontxn_f1'])} |" for sp in EVAL_SPLITS) + f"""

Training path: `{FT_INFO['path']}` · prompt format: `{FT_FORMAT}` · {FT_INFO['n_train']} curated examples · LoRA r={LORA_R}, α={LORA_ALPHA}, lr={LR}, {EPOCHS} epoch(s).

**Licence:** the base model is CC-BY-NC (non-commercial). Commercial deployment needs a licence from Cohere.
**Data:** fully synthetic; bank names are fictional; no real customer data.
"""
''')

code(r'''
if PUSH_TO_HUB:
    # ---- model: LoRA adapter (+ card) ----
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    art = FT_INFO["artifact"]
    open(os.path.join(art, "README.md"), "w").write(MODEL_CARD)
    api.upload_folder(folder_path=art, repo_id=MODEL_REPO, repo_type="model")
    for f in glob.glob(f"{WORK}/results/train_log_*.json") + glob.glob(f"{WORK}/results/train_info_*.json") + glob.glob(f"{WORK}/figures/train_loss_*.png"):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"logs/{os.path.basename(f)}", repo_id=MODEL_REPO)
    print("uploaded", MODEL_REPO, "(adapter + training logs)")
    if RUN_ABLATION:
        api.create_repo(ABL_REPO, repo_type="model", exist_ok=True)
        open(os.path.join(ABL_INFO["artifact"], "README.md"), "w").write(
            f"# Ablation adapter (no hard negatives / noise)\nSee {MODEL_REPO}. Trained on {ABL_INFO['n_train']} clean examples. For comparison only.\n")
        api.upload_folder(folder_path=ABL_INFO["artifact"], repo_id=ABL_REPO, repo_type="model")

    # ---- dataset: all splits, curated train, predictions, metrics, figures ----
    api.create_repo(DATA_REPO, repo_type="dataset", exist_ok=True)
    DATA_CARD = f"""---
license: cc-by-4.0
language: [en, hi]
task_categories: [text-generation, token-classification]
tags: [lexsi-hackathon, synthetic, upi, banking, hinglish, structured-output]
pretty_name: Indian bank & UPI SMS to JSON (synthetic, trilingual)
configs:
- config_name: default
  data_files:
  - {{split: train, path: data/train_curated.jsonl}}
  - {{split: dev, path: data/dev.jsonl}}
  - {{split: test_id, path: data/test_id.jsonl}}
  - {{split: test_ood, path: data/test_ood.jsonl}}
  - {{split: challenge, path: data/challenge_set.jsonl}}
---
# Indian bank & UPI SMS → JSON (English · Hindi · Hinglish)

Synthetic SMS with **exact** labels (a field is labelled only if its text survived generation), plus a 44-example **hand-written challenge set**.
`test_ood` uses message templates that never appear in training. Bank names are fictional; messages contain no links or phone numbers.

Schema: `{json.dumps(FIELDS)}`. Non-transactions (OTP, offers, reminders, future debits, balance info, money requests) → `is_transaction: false`, all other fields null.

Also included: `results/` (every model prediction, metrics, curation report from CuratorKIT, AuditKIT run records) and `figures/`.
"""
    open(f"{WORK}/README_dataset.md", "w").write(DATA_CARD)
    api.upload_file(path_or_fileobj=f"{WORK}/README_dataset.md", path_in_repo="README.md", repo_id=DATA_REPO, repo_type="dataset")
    for sub in ["data", "results", "figures"]:
        api.upload_folder(folder_path=f"{WORK}/{sub}", path_in_repo=sub, repo_id=DATA_REPO, repo_type="dataset")
    if os.path.isdir(f"{WORK}/curated"):
        api.upload_folder(folder_path=f"{WORK}/curated", path_in_repo="curatorkit_run", repo_id=DATA_REPO,
                          repo_type="dataset", allow_patterns=["**/*.json", "**/*.md", "**/*.txt", "**/*.jsonl"])
    print("uploaded", DATA_REPO)
''')

md("### GGUF export for the free CPU demo\nMerge the adapter into the base model, then convert with llama.cpp to `Q8_0` (no compilation needed). The demo Space runs this on 2 free vCPUs.")
code(r'''
GGUF_PATH = None
if MAKE_GGUF:
    try:
        free_gpu()
        from peft import PeftModel
        merged_dir = f"{WORK}/models/merged"
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu")
        art = FT_INFO["artifact"]
        m = PeftModel.from_pretrained(base, art).merge_and_unload() if os.path.exists(f"{art}/adapter_config.json") \
            else AutoModelForCausalLM.from_pretrained(art, torch_dtype=torch.float16, device_map="cpu")
        m.save_pretrained(merged_dir, safe_serialization=True); tok.save_pretrained(merged_dir)
        del m, base; gc.collect()
        if not os.path.isdir(f"{WORK}/llama.cpp"):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", f"{WORK}/llama.cpp"], check=True)
        pip("-r", f"{WORK}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
        GGUF_PATH = f"{WORK}/models/{PROJECT}-q8_0.gguf"
        r = subprocess.run([sys.executable, f"{WORK}/llama.cpp/convert_hf_to_gguf.py", merged_dir,
                            "--outfile", GGUF_PATH, "--outtype", "q8_0"], capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:])
        if r.returncode != 0 or not os.path.exists(GGUF_PATH):
            GGUF_PATH = None
            print("⚠️ GGUF conversion failed; the Space will fall back to transformers on CPU (slower).")
        elif PUSH_TO_HUB:
            api.create_repo(GGUF_REPO, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=GGUF_PATH, path_in_repo=os.path.basename(GGUF_PATH), repo_id=GGUF_REPO)
            open(f"{WORK}/README_gguf.md", "w").write(MODEL_CARD.replace("library_name: peft", "library_name: gguf"))
            api.upload_file(path_or_fileobj=f"{WORK}/README_gguf.md", path_in_repo="README.md", repo_id=GGUF_REPO)
            print("uploaded", GGUF_REPO)
    except Exception:
        traceback.print_exc()
''')

code(r'''
# ---- demo Space (Gradio, CPU) + collections ----
SPACE_FILES_NOTE = "The Space files (app.py, requirements.txt, README.md) are in the repo's space/ folder."
if PUSH_TO_HUB:
    try:
        space_dir = f"{WORK}/space"
        os.makedirs(space_dir, exist_ok=True)
        for f in ["smsgen.py", "scoring.py", "regex_baseline.py"]:
            shutil.copy(f"{WORK}/src/{f}", space_dir)
        open(f"{space_dir}/app.py", "w").write(SPACE_APP)
        open(f"{space_dir}/requirements.txt", "w").write(SPACE_REQS)
        open(f"{space_dir}/README.md", "w").write(SPACE_README.replace("__GGUF_REPO__", GGUF_REPO).replace("__MODEL_REPO__", MODEL_REPO).replace("__DATA_REPO__", DATA_REPO))
        api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
        for k, v in {"GGUF_REPO": GGUF_REPO, "GGUF_FILE": os.path.basename(GGUF_PATH) if GGUF_PATH else "",
                     "LORA_REPO": MODEL_REPO, "BASE_MODEL": BASE_MODEL}.items():
            try: api.add_space_variable(SPACE_REPO, k, v)
            except Exception as e: print("space variable", k, e)
        if token and SPACE_ADD_TOKEN:   # only needed for the slow transformers fallback (gated base model)
            try: api.add_space_secret(SPACE_REPO, "HF_TOKEN", token)
            except Exception as e: print("space secret", e)
        api.upload_folder(folder_path=space_dir, repo_id=SPACE_REPO, repo_type="space")
        print("Space:", f"https://huggingface.co/spaces/{SPACE_REPO}")
    except Exception:
        traceback.print_exc()

    COLLECTIONS = {}
    try:
        cm = api.create_collection(title="Tiny Aya SMS extraction: models", namespace=HF_USERNAME,
                                   description="Lexsi hackathon T1-P5: LoRA + GGUF models", exists_ok=True)
        for rid in [MODEL_REPO] + ([GGUF_REPO] if GGUF_PATH else []) + ([ABL_REPO] if RUN_ABLATION else []):
            api.add_collection_item(cm.slug, item_id=rid, item_type="model", exists_ok=True)
        api.add_collection_item(cm.slug, item_id=SPACE_REPO, item_type="space", exists_ok=True)
        cd = api.create_collection(title="Tiny Aya SMS extraction: datasets", namespace=HF_USERNAME,
                                   description="Lexsi hackathon T1-P5: data, predictions, metrics, logs", exists_ok=True)
        api.add_collection_item(cd.slug, item_id=DATA_REPO, item_type="dataset", exists_ok=True)
        COLLECTIONS = {"models": f"https://huggingface.co/collections/{cm.slug}",
                       "datasets": f"https://huggingface.co/collections/{cd.slug}"}
        print(COLLECTIONS)
    except Exception:
        traceback.print_exc()
''')

md("## 14. Render the Hugging Face article and the abstract from the real numbers\nWrites `ARTICLE.md` and `ABSTRACT.md` into the working directory. Review, add your name, then paste into a Hugging Face article (huggingface.co/new-blog) and the submission form.")
code(r'''
def m(s, sp, k):
    return pct(RESULTS[(s, sp)]["metrics"].get(k)) if (s, sp) in RESULTS else "–"

ctx = {
    "TABLE_TEST_ID": TABLES["test_id"], "TABLE_TEST_OOD": TABLES["test_ood"], "TABLE_CHALLENGE": TABLES["challenge"],
    "LANG_TABLE_OOD": LANG_TABLES["test_ood"], "LANG_TABLE_CHALLENGE": LANG_TABLES["challenge"],
    "TAG_TABLE_OOD": TAG_TABLES["test_ood"], "TAG_TABLE_CHALLENGE": TAG_TABLES["challenge"],
    "SIG_TABLE": SIG_TABLE, "ROI_TABLE": ROI_TABLE,
    "N_TRAIN": str(len(train_rows)), "N_DEV": str(len(splits["dev"])), "N_TEST_ID": str(len(splits["test_id"])),
    "N_TEST_OOD": str(len(splits["test_ood"])), "N_CHALLENGE": str(len(splits["challenge"])),
    "FT_EM_ID": m("fine_tuned", "test_id", "exact_match"), "FT_EM_OOD": m("fine_tuned", "test_ood", "exact_match"),
    "FT_EM_CH": m("fine_tuned", "challenge", "exact_match"), "FS_EM_OOD": m("few_shot", "test_ood", "exact_match"),
    "FS_EM_CH": m("few_shot", "challenge", "exact_match"), "RX_EM_CH": m("regex", "challenge", "exact_match"),
    "ZS_EM_CH": m("zero_shot", "challenge", "exact_match"),
    "FT_HALL_CH": m("fine_tuned", "challenge", "hallucination_rate"), "FS_HALL_CH": m("few_shot", "challenge", "hallucination_rate"),
    "FT_NT_CH": m("fine_tuned", "challenge", "nontxn_f1"), "ABL_NT_CH": m("ft_no_robust_data", "challenge", "nontxn_f1"),
    "ABL_EM_CH": m("ft_no_robust_data", "challenge", "exact_match"), "ABL_HALL_CH": m("ft_no_robust_data", "challenge", "hallucination_rate"),
    "FT_SCHEMA_OOD": m("fine_tuned", "test_ood", "schema_valid"), "ZS_SCHEMA_OOD": m("zero_shot", "test_ood", "schema_valid"),
    "TRAIN_PATH": FT_INFO["path"], "TRAIN_MIN": str(FT_INFO.get("train_minutes")), "FT_FORMAT": FT_FORMAT,
    "CURATION": json.dumps(curation, ensure_ascii=False),
    "WRONG_FIELDS": ", ".join(f"{k} ({v})" for k, v in WRONG_FIELD_COUNTS.most_common(5)) or "none",
    "SELF_COST": f"{ROI['self_hosted']['usd_per_1M_sms']:,.2f}",
    "MONTHLY_SMS": f"{MONTHLY_SMS:,}", "SELF_MONTH": f"{ROI['self_hosted']['usd_per_month_on_demand']:,.2f}",
    "ALWAYS_ON_MONTH": f"{ROI['self_hosted']['usd_per_month_always_on_1gpu']:,.2f}",
    "API_MONTH": ", ".join(f"{k} ${v['usd_per_month']:,.2f}" for k, v in ROI["api"].items()),
    "GPU_HOURS": f"{ROI['self_hosted']['gpu_hours_per_month']:,.1f}", "TOK_S": f"{GEN_TOKENS_PER_SEC:.0f}",
    "TRAIN_COST": f"{ROI['self_hosted']['one_time_training_usd']:.2f}",
    "API_COST": ", ".join(f"{k} ${v['usd_per_1M_sms']:,.2f}" for k, v in ROI["api"].items()),
    "MODEL_REPO": MODEL_REPO, "GGUF_REPO": GGUF_REPO, "DATA_REPO": DATA_REPO, "SPACE_REPO": SPACE_REPO,
    "BASE_MODEL": BASE_MODEL, "LORA_R": str(LORA_R), "LORA_ALPHA": str(LORA_ALPHA), "LR": str(LR), "EPOCHS": str(EPOCHS),
}
def render_tpl(t):
    for k, v in ctx.items():
        t = t.replace("{{" + k + "}}", v)
    return t
open(f"{WORK}/ARTICLE.md", "w", encoding="utf-8").write(render_tpl(ARTICLE_TEMPLATE))
open(f"{WORK}/ABSTRACT.md", "w", encoding="utf-8").write(render_tpl(ABSTRACT_TEMPLATE))
abstract_words = len(render_tpl(ABSTRACT_TEMPLATE).split())
print("ARTICLE.md and ABSTRACT.md written | abstract words:", abstract_words, "(limit 500)")
print(render_tpl(ABSTRACT_TEMPLATE))
''')

# placeholders cell for templates & space files - inserted BEFORE push cells by build step
TEMPLATES_CELL = (
    f"ARTICLE_TEMPLATE = {read('docs/ARTICLE_TEMPLATE.md')!r}\n"
    f"ABSTRACT_TEMPLATE = {read('docs/ABSTRACT_TEMPLATE.md')!r}\n"
    f"SPACE_APP = {read('space/app.py')!r}\n"
    f"SPACE_REQS = {read('space/requirements.txt')!r}\n"
    f"SPACE_README = {read('space/README.md')!r}\n"
    "print('templates loaded')"
)
# insert right after the source-files cell (index of the challenge cell + 1)
idx = next(i for i, c in enumerate(cells) if "CHALLENGE_JSONL" in c["source"]) + 1
cells.insert(idx, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": "# ---- article/abstract templates and demo Space files ----\n" + TEMPLATES_CELL})

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "accelerator": "GPU", "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "sms_extraction_pipeline.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
