"""Builds notebooks/stage2_frontier.ipynb: big closed and open LLMs on the same test sets, same scorer."""
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
# Stage 2 · Frontier LLMs on the same tests

**Tiny Aya SMS Extract** · Anshul J Amin · Lexsi AI Safety & Alignment Hackathon, Track 1

Can a big general model do this task without fine-tuning? We run **Cohere Command A** (closed, Cohere API) and **Llama 3.3 70B Instruct** (open, Hugging Face Inference Providers) on exactly the same messages as our fine-tuned Tiny Aya: seen formats, unseen formats, hand-written, their rephrased versions and the unrelated texts.

Fair setup: the same system prompt with every schema rule, the same three few-shot examples Tiny Aya's few-shot baseline used, JSON mode on, temperature 0. Scoring: the same scorer and **AuditKIT**. We also measure tokens (for cost per 1M SMS) and latency, and compare with Tiny Aya on the same message ids (McNemar).

**How to run (Kaggle):** *File → Import Notebook* → this file. **Internet ON**, no accelerator. Secrets: `COHERE_API_KEY`, `HF_TOKEN`. Run `QUICK_MODE = True` first, then `QUICK_MODE = False`.
""")

code(r'''
import os, sys, json, time, random, subprocess, threading, re
import concurrent.futures as cf
from collections import Counter

QUICK_MODE = True
SEED = 7
V3_DATA_REPO = "Meowniece/tiny-aya-sms-extract-v3-data"
TINY_AYA_RESULTS_REPO = "Meowniece/tiny-aya-sms-extract-fire-v3-data"   # written by stage2_train_eval.ipynb

SPLITS = ["test_id", "test_ood", "challenge", "test_id_rephrased", "test_ood_rephrased", "challenge_rephrased", "unrelated"]
SETTINGS = ["few_shot"]          # add "zero_shot" if your budget allows (doubles the calls)

RUN_COMMAND = True
COMMAND_PREFERENCE = ["command-a-plus-05-2026", "command-a-03-2025"]   # Cohere flagship, auto-detected below
COMMAND_RPM = 18                                 # trial keys: 20 calls/min and ~1,000 calls/month (not enough for all
                                                 # 1,700 messages: use a production key, or limit SPLITS)
RUN_OPEN = False          # needs Hugging Face Inference Providers credit (HF returned 402 without it)
OPEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # Hugging Face Inference Providers
OPEN_RPM = 60
HF_BILL_TO = ""                                   # e.g. the hackathon org name, to bill its credits; "" = your account

# USD per 1M tokens (input, output). Check the current list prices before reporting.
PRICES = {"command_a": (2.50, 10.00), "llama_3_3_70b": (0.88, 0.88)}   # Command A list price; update if Command A Plus differs

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work_frontier")
for d in ["src", "data", "results", "cache"]:
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
N_PER_SPLIT = 6 if QUICK_MODE else None
print("work dir:", WORK, "| quick mode:", QUICK_MODE)
''')

md("## 1. Install and read secrets")
code(r'''
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + " ".join(args))
    if r.returncode: print(r.stderr[-1500:])
    return r.returncode == 0
OK_CURATOR = pip("curatorkit")
OK_AUDIT = pip("auditkit")
pip("litellm", "huggingface_hub", "requests")

def secret(name):
    if os.environ.get(name): return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None
for k in ["COHERE_API_KEY", "HF_TOKEN"]:
    v = secret(k)
    if v: os.environ[k] = v
    print(k, "found" if v else "MISSING")
''')

md("## 2. Project source files")
for path in ["src/smsgen.py", "src/scoring.py", "src/stage2_data.py", "src/regex_baseline.py"]:
    content = read(path)
    code(f"# ---- {path} ----\nCONTENT = {content!r}\n"
         f"open(f\"{{WORK}}/{path}\", \"w\", encoding=\"utf-8\").write(CONTENT)\n"
         f"print('wrote {path}', len(CONTENT), 'chars')")
code(r'''
sys.path.insert(0, f"{WORK}/src")
import importlib, smsgen, scoring as S, stage2_data as D
for m in (smsgen, S, D): importlib.reload(m)
from smsgen import gold_to_str, build_messages, SYSTEM_PROMPT
''')

md("## 3. Data and the few-shot examples\nThe few-shot examples are picked with exactly the same code and seed as Tiny Aya's few-shot baseline in `stage2_train_eval.ipynb`.")
code(r'''
from huggingface_hub import hf_hub_download, HfApi, login
if os.environ.get("HF_TOKEN"): login(token=os.environ["HF_TOKEN"])
def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
files = {"train": "train_v3", **{s: s for s in SPLITS}}
splits = {k: load_jsonl(hf_hub_download(V3_DATA_REPO, f"data/{v}.jsonl", repo_type="dataset")) for k, v in files.items()}
if N_PER_SPLIT:
    splits = {k: (v if k == "train" else v[:N_PER_SPLIT]) for k, v in splits.items()}
for r in splits["unrelated"]: r.setdefault("tags", ["unrelated"])
train_rows = splits["train"]
_rng = random.Random(SEED)
def _pick(lang, txn):
    pool = [r for r in train_rows if r["lang"] == lang and r["gold"]["is_transaction"] == txn] \
        or [r for r in train_rows if r["gold"]["is_transaction"] == txn]
    clean = [r for r in pool if set(r["tags"]) <= {"hard_negative"}]
    return _rng.choice(clean or pool)
FEW_SHOT = [(r["sms"], gold_to_str(r["gold"])) for r in [_pick("en", True), _pick("hi", True), _pick("hinglish", False)]]
for s, g in FEW_SHOT: print("example:", s[:90], "->", g[:80])
print({k: len(v) for k, v in splits.items()})
''')

md("## 4. Models (via CuratorKIT's `LiteLLMBackend`)")
code(r'''
import requests
from curatorkit.llm.litellm import LiteLLMBackend
def cohere_models():
    try:
        r = requests.get("https://api.cohere.com/v1/models", params={"endpoint": "chat", "page_size": 1000},
                         headers={"Authorization": f"Bearer {os.environ.get('COHERE_API_KEY', '')}"}, timeout=30)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print("could not list Cohere models:", e); return []
names = cohere_models()
cmd = next((p for p in COMMAND_PREFERENCE if p in names), None) or next(
    (n for n in names if n.startswith("command-a") and not any(x in n for x in ["vision", "reasoning", "translate"])), COMMAND_PREFERENCE[0])
MODELS = {}
if RUN_COMMAND:
    MODELS["command_a"] = {"backend": LiteLLMBackend(model=f"cohere_chat/{cmd}", temperature=0.0, max_tokens=300, max_retries=6),
                           "rpm": COMMAND_RPM, "name": cmd}
if RUN_OPEN:
    MODELS["llama_3_3_70b"] = {"backend": LiteLLMBackend(model=f"openai/{OPEN_MODEL}", api_base="https://router.huggingface.co/v1",
                                                         api_key=os.environ.get("HF_TOKEN"), temperature=0.0, max_tokens=300, max_retries=4),
                               "rpm": OPEN_RPM, "name": OPEN_MODEL}
EXTRA = {"extra_headers": {"X-HF-Bill-To": HF_BILL_TO}} if HF_BILL_TO else {}
for k, m in MODELS.items():
    try:
        kw = EXTRA if k == "llama_3_3_70b" else {}
        out = m["backend"].generate([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=5, **kw)
        print(k, m["name"], "-> probe OK:", out.text.strip()[:20])
    except Exception as e:
        print(k, m["name"], "-> probe FAILED:", str(e)[:300])
n_calls = sum(len(splits[s]) for s in SPLITS) * len(SETTINGS)
print(f"calls per model: {n_calls}")
''')

md("## 5. Run (cached, rate-limited; JSON mode on)")
code(r'''
class RateLimiter:
    def __init__(self, rpm):
        self.dt, self.lock, self.next = 60.0 / max(rpm, 1), threading.Lock(), time.time()
    def wait(self):
        with self.lock:
            t = max(self.next, time.time()); self.next = t + self.dt
        time.sleep(max(0.0, t - time.time()))

def run_llm(task, items, make_messages, backend, rpm, workers=6, **gen_kwargs):
    path = f"{WORK}/cache/{task}.jsonl"
    cache = {}
    if os.path.exists(path):
        for rec in load_jsonl(path):
            if not rec.get("error"): cache[rec["id"]] = rec
    todo = [it for it in items if it["id"] not in cache]
    print(f"[{task}] cached {len(items) - len(todo)}, calling {len(todo)}")
    lim, lock, t0 = RateLimiter(rpm), threading.Lock(), time.time()
    fh = open(path, "a", encoding="utf-8")
    state = {"json_mode": True}
    def one(it):
        lim.wait()
        kw = dict(gen_kwargs)
        if state["json_mode"]: kw["response_format"] = {"type": "json_object"}
        t1 = time.time()
        try:
            r = backend.generate(make_messages(it), **kw)
            rec = {"id": it["id"], "text": r.text, "tokens_in": r.prompt_tokens, "tokens_out": r.completion_tokens,
                   "latency_s": round(time.time() - t1, 3), "json_mode": "response_format" in kw, "error": None}
        except Exception as e:
            msg = str(e)
            if "response_format" in msg or "json" in msg.lower() and "support" in msg.lower():
                state["json_mode"] = False
            rec = {"id": it["id"], "text": "", "error": f"{type(e).__name__}: {msg[:300]}"}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            if not rec["error"]: cache[it["id"]] = rec
        return rec
    errors = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            errors += bool(rec["error"])
            if i % 100 == 0 or i == len(todo): print(f"  {i}/{len(todo)}, {errors} errors, {time.time() - t0:.0f}s")
    fh.close()
    if errors: print("  re-run this cell to retry only the failed calls")
    return cache

PROMPTS = {"zero_shot": lambda r: build_messages(r["sms"]), "few_shot": lambda r: build_messages(r["sms"], FEW_SHOT)}
RAW = {}
for mk, m in MODELS.items():
    for setting in SETTINGS:
        for sp in SPLITS:
            kw = EXTRA if mk == "llama_3_3_70b" else {}
            RAW[(mk, setting, sp)] = run_llm(f"{mk}_{setting}_{sp}", splits[sp], PROMPTS[setting], m["backend"], m["rpm"], **kw)
''')

md("## 6. Score with the same scorer and AuditKIT")
code(r'''
ak = None
if OK_AUDIT:
    try:
        import auditkit as ak
    except Exception as e:
        print("AuditKIT import failed:", e)
RESULTS = {}
def pct(x): return "–" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.1f}"
for (mk, setting, sp), cache in RAW.items():
    rows = [r for r in splits[sp] if r["id"] in cache]
    outs = [cache[r["id"]]["text"] for r in rows]
    scored = [S.score_example(o, r["gold"]) for o, r in zip(outs, rows)]
    audit = {"auditkit": False}
    if ak is not None and rows:
        try:
            by_sms = {r["sms"]: o for r, o in zip(rows, outs)}
            res = ak.evaluate([ak.Sample(input=r["sms"], target=gold_to_str(r["gold"])) for r in rows],
                              model=lambda prompts: [by_sms.get(p, "") for p in prompts])
            audit = {"auditkit": True, "headline": str(getattr(res, "headline", ""))[:500]}
        except Exception as e:
            audit["auditkit_error"] = str(e)[:300]
    name = f"{mk}_{setting}"
    RESULTS[(name, sp)] = {"rows": rows, "outputs": outs, "scored": scored, "metrics": S.aggregate(scored),
                           "missing": len(splits[sp]) - len(rows), "audit": audit,
                           "tokens_in": [cache[r["id"]].get("tokens_in", 0) for r in rows],
                           "tokens_out": [cache[r["id"]].get("tokens_out", 0) for r in rows],
                           "latency": [cache[r["id"]].get("latency_s") for r in rows]}
    write_jsonl(f"{WORK}/results/predictions_{name}_{sp}.jsonl",
                [{"id": r["id"], "sms": r["sms"], "gold": r["gold"], "output": o, "exact_match": s["exact_match"]}
                 for r, o, s in zip(rows, outs, scored)])

KEYS = [("exact_match", "Exact match"), ("field_acc_txn", "Field acc."), ("hallucination_rate", "Halluc. ↓"),
        ("omission_rate", "Omission ↓"), ("nontxn_f1", "Non-txn F1"), ("schema_valid", "Schema valid")]
lines = []
for sp in SPLITS:
    lines += [f"\n### {sp}", "| System | n | EM [95% CI] | " + " | ".join(n for k, n in KEYS[1:]) + " |", "|---|---|---|" + "---|" * (len(KEYS) - 1)]
    for (name, s2), v in RESULTS.items():
        if s2 != sp: continue
        ci = S.bootstrap_ci([float(x["exact_match"]) for x in v["scored"]], seed=SEED)
        m = v["metrics"]
        lines.append(f"| {name} | {len(v['rows'])} | {pct(m.get('exact_match'))} [{pct(ci[0])}–{pct(ci[1])}] | "
                     + " | ".join(pct(m.get(k)) for k, _ in KEYS[1:]) + " |")
FRONTIER_TABLE = "\n".join(lines)
print(FRONTIER_TABLE)
''')

md("## 7. Cost and latency")
code(r'''
cost_lines = ["| System | Avg input tokens | Avg output tokens | USD per 1M SMS (list price) | Median latency (s) |", "|---|---|---|---|---|"]
COST = {}
for mk in MODELS:
    for setting in SETTINGS:
        name = f"{mk}_{setting}"
        tin = [t for (n, sp), v in RESULTS.items() if n == name for t in v["tokens_in"] if t]
        tout = [t for (n, sp), v in RESULTS.items() if n == name for t in v["tokens_out"] if t]
        lat = sorted(l for (n, sp), v in RESULTS.items() if n == name for l in v["latency"] if l is not None)
        if not tin: continue
        ai, ao = sum(tin) / len(tin), sum(tout) / max(len(tout), 1)
        pin, pout = PRICES.get(mk, (float("nan"), float("nan")))
        usd = (ai * pin + ao * pout)   # per 1M SMS: tokens per SMS x USD per 1M tokens
        COST[name] = {"avg_in": ai, "avg_out": ao, "usd_per_1M_sms": usd, "median_latency_s": lat[len(lat) // 2] if lat else None}
        cost_lines.append(f"| {name} | {ai:.0f} | {ao:.0f} | {usd:,.2f} | {COST[name]['median_latency_s']} |")
COST_TABLE = "\n".join(cost_lines)
print(COST_TABLE)
print("Latency is wall-clock per API call from Kaggle with several calls in flight; treat it as indicative.")
''')

md("## 8. Paired comparison with the fine-tuned Tiny Aya (same message ids)\nNeeds the predictions from `stage2_train_eval.ipynb`. If that run is not finished yet, run this cell again later.")
code(r'''
PAIR_TABLE = ""
try:
    lines = ["| Tiny Aya system | Frontier system | Set | Tiny Aya EM | Frontier EM | Tiny Aya-only correct | Frontier-only correct | McNemar p |",
             "|---|---|---|---|---|---|---|---|"]
    for ta in ["fine_tuned", "fire_v2"]:
        for sp in SPLITS:
            p = hf_hub_download(TINY_AYA_RESULTS_REPO, f"results/predictions_{ta}_{sp}.jsonl", repo_type="dataset")
            ta_em = {r["id"]: bool(r["exact_match"]) for r in load_jsonl(p)}
            for (name, s2), v in RESULTS.items():
                if s2 != sp: continue
                fr = {r["id"]: bool(x["exact_match"]) for r, x in zip(v["rows"], v["scored"])}
                ids = [i for i in fr if i in ta_em]
                if not ids: continue
                mc = S.mcnemar_exact([fr[i] for i in ids], [ta_em[i] for i in ids])
                lines.append(f"| {ta} | {name} | {sp} | {pct(sum(ta_em[i] for i in ids) / len(ids))} | {pct(sum(fr[i] for i in ids) / len(ids))} | "
                             f"{mc['b_only']} | {mc['a_only']} | {mc['p_value']:.2g} |")
    PAIR_TABLE = "\n".join(lines)
    print(PAIR_TABLE)
except Exception as e:
    print("Tiny Aya predictions not available yet:", str(e)[:200])
''')

md("## 9. Save and upload (into the v3 dataset repo, `results/frontier/`)")
code(r'''
summary = {"models": {k: m["name"] for k, m in MODELS.items()}, "settings": SETTINGS, "prices_usd_per_1M_tokens": PRICES,
           "metrics": {f"{n}/{sp}": {**v["metrics"], "missing": v["missing"], "audit": v["audit"]} for (n, sp), v in RESULTS.items()},
           "cost": COST}
json.dump(summary, open(f"{WORK}/results/frontier_summary.json", "w"), indent=1, default=str)
open(f"{WORK}/results/frontier_tables.md", "w", encoding="utf-8").write(
    "## Frontier LLMs\n" + FRONTIER_TABLE + "\n\n## Cost and latency\n" + COST_TABLE + "\n\n## Paired with Tiny Aya\n" + PAIR_TABLE)
if not QUICK_MODE:
    HfApi().upload_folder(folder_path=f"{WORK}/results", path_in_repo="results/frontier", repo_id=V3_DATA_REPO,
                          repo_type="dataset", commit_message="Frontier LLM baselines (Command A, Llama 3.3 70B)")
    print("uploaded to", V3_DATA_REPO, "results/frontier/")
else:
    print("QUICK_MODE: not uploaded")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "none", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "stage2_frontier.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
