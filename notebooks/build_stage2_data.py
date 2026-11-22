"""Builds notebooks/stage2_data.ipynb: data v3 (LLM rewrites), rephrased test sets, unrelated-text set."""
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
# Stage 2 · Data: more varied training SMS, rephrased test sets, unrelated texts

**Tiny Aya SMS Extract** · Anshul J Amin · Lexsi AI Safety & Alignment Hackathon, Track 1

This notebook builds three new datasets. No GPU is needed.

1. **Data v3 (training):** every Stage 1 training SMS is rewritten once by **Aya Expanse 32B** (Cohere), called through **CuratorKIT**'s LLM backend. The gold label is reused unchanged, so a rewrite is kept only if it passes four checks: hygiene, language, **facts** (every labelled value is still in the text and no new number appears) and **meaning** (a rule-based reading agrees on transaction / debit-credit / status). The combined set then goes through **CuratorKIT** exact de-duplication, our JSON gate, CuratorKIT's **diversity gate** (near-duplicates) and a grounding check. Counts are logged after every step.
2. **Rephrased test sets:** all 644 test SMS (seen formats, unseen formats, hand-written) are paraphrased by a *different* model that is not used for training data or as a compared system, with the same fact and meaning checks.
3. **Unrelated texts:** personal and spam SMS (UCI SMS Spam Collection) and general English and Hindi sentences (Wikipedia). Correct output: not a transaction, every field empty.

**How to run (Kaggle):** *File → Import Notebook* → this file. Settings: **Internet ON** (no accelerator needed). Add-ons → Secrets: `COHERE_API_KEY` and `HF_TOKEN` (a Hugging Face *write* token). Run with `QUICK_MODE = True` first (about 5 minutes), then `QUICK_MODE = False`.
""")

code(r'''
import os, sys, json, time, random, subprocess, threading, glob, inspect, re
import concurrent.futures as cf
from collections import Counter, defaultdict

QUICK_MODE = True          # True = a few rows per step (smoke test). Then set False and run all.
PUSH_TO_HUB = True         # upload the v3 dataset and add it to the dataset collection
SEED = 7

V2_DATA_REPO   = "Meowniece/tiny-aya-sms-extract-v2-data"
V3_DATA_REPO   = "Meowniece/tiny-aya-sms-extract-v3-data"
DATA_COLLECTION = "Meowniece/tiny-aya-sms-extraction-datasets-6afa2f2876dd2d18a69a3e77"

# LLM access. True = Hugging Face Inference Providers, billed to the hackathon organisation's credits.
# False = Cohere API with COHERE_API_KEY (a trial key allows ~1,000 calls/month).
USE_HF_ROUTER = True
HF_BILL_TO = "tinyaya-sms-extract"        # HF organisation whose credits pay for the calls
REWRITER_HF = "CohereLabs/aya-expanse-32b:cohere"   # Aya Expanse 32B, served by Cohere through HF
# Training rewrites via the Cohere API (used only when USE_HF_ROUTER = False)
REWRITER_PREFERENCE = ["c4ai-aya-expanse-32b", "aya-expanse-32b"]
N_REWRITE = 1600           # training SMS to rewrite (1 call each). A Cohere *trial* key allows ~1,000 calls/month:
                           # with a trial key set this to 700 so that calls remain for the rest of the work.
COHERE_RPM = 18            # trial keys: 20 chat calls/minute. A production key can use 300+.
HF_ROUTER_RPM = 60         # Hugging Face Inference Providers

# Test rephrasing: a model that is NOT the training rewriter and NOT a compared system.
TEST_REPHRASER_HF = "Qwen/Qwen2.5-72B-Instruct"          # via Hugging Face Inference Providers (HF_TOKEN)
TEST_REPHRASER_FALLBACK = ["command-r-plus-08-2024", "command-r-08-2024", "command-r7b-12-2024"]   # Cohere, used if HF fails;
                                                   # never Command A (a compared system) or Aya Expanse (the training rewriter)
HF_RPM = 60   # test rephraser rate (HF Inference Providers)

DIVERSITY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIVERSITY_THRESHOLD = 0.97  # cosine similarity at or above this = near-duplicate
N_UNRELATED = {"sms_ham": 125, "sms_spam": 125, "wiki_en": 125, "wiki_hi": 125}

if QUICK_MODE:
    N_REWRITE, N_TEST_PER_SPLIT = 24, 8
    N_UNRELATED = {k: 5 for k in N_UNRELATED}
else:
    N_TEST_PER_SPLIT = None   # all

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work_stage2")
for d in ["src", "data", "results", "cache", "curated"]:
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
random.seed(SEED)
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
pip("litellm", "sentence-transformers", "faiss-cpu", "datasets", "huggingface_hub", "requests")

def secret(name):
    if os.environ.get(name): return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

for k in ["COHERE_API_KEY", "HF_TOKEN"]:
    v = secret(k)
    if v: os.environ[k] = v
    print(k, "found" if v else "MISSING")
import curatorkit
print("curatorkit", curatorkit.__version__)
''')

md("## 2. Project source files")
for path in ["src/smsgen.py", "src/scoring.py", "src/regex_baseline.py", "src/stage2_data.py"]:
    content = read(path)
    code(f"# ---- {path} ----\nCONTENT = {content!r}\n"
         f"open(f\"{{WORK}}/{path}\", \"w\", encoding=\"utf-8\").write(CONTENT)\n"
         f"print('wrote {path}', len(CONTENT), 'chars')")
code(r'''
sys.path.insert(0, f"{WORK}/src")
import importlib, smsgen, scoring as S, regex_baseline as RB, stage2_data as D
for m in (smsgen, S, RB, D): importlib.reload(m)
from smsgen import gold_to_str, SYSTEM_PROMPT
''')

md("## 3. Load the Stage 1 (v2) data")
code(r'''
from huggingface_hub import hf_hub_download, HfApi, login, whoami
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
    print("HF user:", whoami()["name"])

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

V2 = {}
for name in ["train", "dev", "test_id", "test_ood", "challenge"]:
    V2[name] = load_jsonl(hf_hub_download(V2_DATA_REPO, f"data/{name}.jsonl", repo_type="dataset"))
V2_CURATION = json.load(open(hf_hub_download(V2_DATA_REPO, "results/curation_report.json", repo_type="dataset")))
print({k: len(v) for k, v in V2.items()})
ALL_EXISTING = {D.norm(r["sms"]).casefold() for rows in V2.values() for r in rows}
''')

md("## 4. LLM backends (CuratorKIT `LiteLLMBackend`)\nBy default every call goes through Hugging Face Inference Providers and is billed to the hackathon organisation (`X-HF-Bill-To`). The training rewriter is Aya Expanse 32B (served by Cohere); the test rephraser is Qwen2.5-72B. Both are probed first, and the run stops early if the rewriter is unreachable.")
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

def pick(names, preferred, contains):
    for p in preferred:
        if p in names: return p
    for n in names:
        if all(c in n for c in contains): return n
    return None

COHERE_NAMES = cohere_models() if not USE_HF_ROUTER else []
EXTRA = {"extra_headers": {"X-HF-Bill-To": HF_BILL_TO}} if (USE_HF_ROUTER and HF_BILL_TO) else {}
def hf_backend(model, temperature):
    return LiteLLMBackend(model=f"openai/{model}", api_base="https://router.huggingface.co/v1",
                          api_key=os.environ.get("HF_TOKEN"), temperature=temperature, max_tokens=300, max_retries=4)

def probe(backend, extra=None):
    try:
        out = backend.generate([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=5, **(extra or {}))
        return bool(out.text.strip()), out.text.strip()[:40]
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:300]}"

if USE_HF_ROUTER:
    REWRITER, REWRITE_RPM, RW_EXTRA = REWRITER_HF, HF_ROUTER_RPM, EXTRA
    rewriter = hf_backend(REWRITER_HF, 0.9)
else:
    print("Cohere chat models:", COHERE_NAMES)
    REWRITER = pick(COHERE_NAMES, REWRITER_PREFERENCE, ["aya-expanse-32b"]) or pick(COHERE_NAMES, [], ["aya-expanse"]) \
               or REWRITER_PREFERENCE[0]
    REWRITE_RPM, RW_EXTRA = COHERE_RPM, {}
    rewriter = LiteLLMBackend(model=f"cohere_chat/{REWRITER}", temperature=0.9, max_tokens=300, max_retries=6)
ok, msg = probe(rewriter, RW_EXTRA)
print("training rewriter:", REWRITER, "| billed to:", HF_BILL_TO if RW_EXTRA else "Cohere key", "| probe:", ok, msg)
if not ok:
    raise RuntimeError("The training rewriter is not reachable; fix access before spending a run: " + msg)

rephraser = hf_backend(TEST_REPHRASER_HF, 0.7)
REPHRASER, REPHRASER_RPM, RP_EXTRA = TEST_REPHRASER_HF, HF_RPM, EXTRA
ok, msg = probe(rephraser, RP_EXTRA)
if not ok:
    COHERE_NAMES = COHERE_NAMES or cohere_models()
    fb = pick(COHERE_NAMES, TEST_REPHRASER_FALLBACK, ["command-r"]) or TEST_REPHRASER_FALLBACK[0]
    print(f"HF Inference Providers unavailable ({msg}) -> test rephraser falls back to Cohere {fb}")
    rephraser = LiteLLMBackend(model=f"cohere_chat/{fb}", temperature=0.7, max_tokens=300, max_retries=6)
    REPHRASER, REPHRASER_RPM, RP_EXTRA = fb, COHERE_RPM, {}
REPHRASE_OK, _msg = probe(rephraser, RP_EXTRA)
print("test rephraser:", REPHRASER, "| probe:", REPHRASE_OK, _msg)
if not REPHRASE_OK:
    print("⚠️ no working test rephraser: the rephrased test sets will be skipped (everything else still runs)")
''')

md("## 5. Rate-limited, cached LLM runner\nEvery response is appended to a cache file, so a re-run (or a crash) never pays for the same call twice.")
code(r'''
class RateLimiter:
    def __init__(self, rpm):
        self.dt, self.lock, self.next = 60.0 / max(rpm, 1), threading.Lock(), time.time()
    def wait(self):
        with self.lock:
            t = max(self.next, time.time()); self.next = t + self.dt
        time.sleep(max(0.0, t - time.time()))

def run_llm(task, items, make_messages, backend, rpm, temperature, workers=6, **gen_kwargs):
    path = f"{WORK}/cache/{task}.jsonl"
    cache = {}
    if os.path.exists(path):
        for rec in load_jsonl(path):
            if not rec.get("error") and rec.get("text"): cache[rec["id"]] = rec
    todo = [it for it in items if it["id"] not in cache]
    print(f"[{task}] cached {len(items) - len(todo)}, calling {len(todo)} ({backend.model}, ~{len(todo) / max(rpm, 1):.0f} min)")
    lim, lock, t0 = RateLimiter(rpm), threading.Lock(), time.time()
    fh = open(path, "a", encoding="utf-8")
    def one(it):
        lim.wait()
        try:
            r = backend.generate(make_messages(it), temperature=temperature, **gen_kwargs)
            rec = {"id": it["id"], "text": r.text, "model": backend.model, "tokens_in": r.prompt_tokens,
                   "tokens_out": r.completion_tokens, "error": None}
        except Exception as e:
            rec = {"id": it["id"], "text": "", "model": backend.model, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            if not rec["error"]: cache[it["id"]] = rec
        return rec
    errors = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            errors += bool(rec["error"])
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {errors} errors, {time.time() - t0:.0f}s")
    fh.close()
    if errors:
        errs = Counter(r["error"].split(":")[0] for r in load_jsonl(path) if r.get("error"))
        print("  error types:", dict(errs), "| re-run this cell to retry only the failed calls")
    return {k: v for k, v in cache.items() if k in {it["id"] for it in items}}
''')

md("## 6. Rewrite the training SMS (Aya Expanse 32B) and check every rewrite")
code(r'''
rng = random.Random(SEED)
src_rows = list(V2["train"]); rng.shuffle(src_rows)
src_rows = src_rows[:N_REWRITE]
make_rw = lambda r: D.train_rewrite_messages(r, random.Random(f"{SEED}-{r['id']}"))
rw_out = run_llm("train_rewrites", src_rows, make_rw, rewriter, REWRITE_RPM, temperature=0.9, workers=8, **RW_EXTRA)

COUNTS = {"v3": {}}
rewrite_log, accepted, existing = [], [], set(ALL_EXISTING)
reasons = Counter()
for r in src_rows:
    rec = rw_out.get(r["id"])
    if rec is None:
        why, txt = "api_error", ""
    else:
        txt = D.clean_llm_text(rec["text"])
        ok, why = D.check_rewrite(r, txt, existing) if txt else (False, "empty")
    reasons[why] += 1
    rewrite_log.append({"id": f"{r['id']}-rw", "source_id": r["id"], "lang": r["lang"], "family": r["family"],
                        "orig_sms": r["sms"], "rewrite": txt, "accepted": why == "ok", "reason": why,
                        "model": REWRITER})
    if why == "ok":
        existing.add(D.norm(txt).casefold())
        accepted.append({**r, "id": f"{r['id']}-rw", "sms": txt, "source_id": r["id"],
                         "template_id": r["template_id"] + "+rw", "tags": list(r.get("tags", [])) + ["llm_rewrite"],
                         "rewritten_by": REWRITER})
write_jsonl(f"{WORK}/data/rewrites_log.jsonl", rewrite_log)
COUNTS["v3"]["rewrite_attempts"] = len(src_rows)
COUNTS["v3"]["rewrite_reasons"] = dict(reasons)
COUNTS["v3"]["rewrites_accepted"] = len(accepted)
print("rewrite outcomes:", dict(reasons.most_common()))
print(f"accepted {len(accepted)}/{len(src_rows)} ({len(accepted) / max(len(src_rows), 1):.0%})",
      "| by language:", dict(Counter(a["lang"] for a in accepted)))
for x in random.Random(1).sample([l for l in rewrite_log if l["accepted"]], min(6, len(accepted))):
    print(f"\n[{x['lang']}/{x['family']}]\n  original: {x['orig_sms']}\n  rewrite : {x['rewrite']}")
''')

md("## 7. CuratorKIT on the combined set: exact de-dup → JSON gate → diversity gate → grounding check\nThe SMS goes into the `instruction` field, because CuratorKIT's exact de-duplication compares instruction + output (in Stage 1 the SMS sat in `input`, so identical all-null labels collapsed and were restored afterwards).")
code(r'''
from curatorkit import Curator, CuratorConfig

def make_config(cls, **kw):
    try:
        params = inspect.signature(cls).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        pass
    return cls(**kw)

combined = list(V2["train"]) + accepted
COUNTS["v3"]["combined_in"] = len(combined)
key = lambda r: f"{SYSTEM_PROMPT}\n\nSMS: {r['sms']}"
by_key = {key(r): r for r in combined}
raw_path = f"{WORK}/data/train_v3_alpaca_raw.jsonl"
write_jsonl(raw_path, [{"instruction": key(r), "input": "", "output": gold_to_str(r["gold"]), "id": r["id"]} for r in combined])

out_dir = f"{WORK}/curated/v3_exact"
cur = Curator(make_config(CuratorConfig, dataset=raw_path, dedup="exact", clean=False,
                          export_formats=["alpaca"], output_dir=out_dir)).run()
try: cur.print_summary()
except Exception: pass
COUNTS["v3"]["curatorkit_stage_counts"] = getattr(cur, "stage_counts", {})
exported = []
for f in glob.glob(f"{out_dir}/**/*alpaca*.json*", recursive=True):
    exported += load_jsonl(f) if f.endswith(".jsonl") else json.load(open(f))
COUNTS["v3"]["after_curatorkit_exact_dedup"] = len(exported)

after_gate, seen = [], set()
for e in exported:
    r = by_key.get(e.get("instruction", ""))
    if r is None or r["id"] in seen: continue
    if S.extract_json(e.get("output", "")) is not None and S.score_example(e["output"], r["gold"])["exact_match"]:
        after_gate.append(r); seen.add(r["id"])
COUNTS["v3"]["after_json_gate"] = len(after_gate)

# CuratorKIT diversity gate: originals first, so a rewrite is dropped if it is a near-copy of anything kept before it.
from curatorkit.schema import DataSample
from curatorkit.gates.diversity import DiversityGate
gate = DiversityGate(embedding_model=DIVERSITY_MODEL, similarity_threshold=DIVERSITY_THRESHOLD, text_field="instruction")
ordered = sorted(after_gate, key=lambda r: "-rw" in r["id"])
samples = [DataSample(source_uri="v3", instruction=r["sms"], output=gold_to_str(r["gold"]), metadata={"id": r["id"]}) for r in ordered]
passed, rejected = gate.run(samples)
rej_ids = {s.metadata["id"] for s in rejected}
COUNTS["v3"]["diversity_flagged_originals_kept"] = sum(1 for i in rej_ids if "-rw" not in i)
COUNTS["v3"]["diversity_rejected_rewrites"] = sum(1 for i in rej_ids if "-rw" in i)
after_div = [r for r in after_gate if not ("-rw" in r["id"] and r["id"] in rej_ids)]
COUNTS["v3"]["after_diversity_gate"] = len(after_div)

# Grounding (hallucination) check: every labelled value must be found in the SMS text.
train_v3 = [r for r in after_div if D.fact_gate(r["sms"], r["sms"], r["gold"])[0]]
COUNTS["v3"]["after_grounding_check"] = len(train_v3)
for r in train_v3: r["split"] = "train"
write_jsonl(f"{WORK}/data/train_v3.jsonl", train_v3)
print(json.dumps({k: v for k, v in COUNTS["v3"].items() if k != "curatorkit_stage_counts"}, indent=1))
print("final v3 train:", len(train_v3), "| by language:", dict(Counter(r["lang"] for r in train_v3)),
      "| rewrites:", sum("-rw" in r["id"] for r in train_v3))
''')

md("## 8. Rephrased test sets (different model, same checks)")
code(r'''
train_texts = {D.norm(r["sms"]).casefold() for r in train_v3} | {D.norm(r["sms"]).casefold() for r in V2["dev"]}
REPHRASED, COUNTS["rephrase"] = {}, {}
for split in (["test_id", "test_ood", "challenge"] if REPHRASE_OK else []):
    rows = V2[split][:N_TEST_PER_SPLIT] if N_TEST_PER_SPLIT else V2[split]
    out1 = run_llm(f"rephrase_{split}", rows, D.test_rephrase_messages, rephraser, REPHRASER_RPM, temperature=0.7, workers=8, **RP_EXTRA)
    kept, reasons, failed = {}, Counter(), []
    def try_accept(r, rec, round_):
        txt = D.clean_llm_text(rec["text"]) if rec else ""
        ok, why = D.check_rewrite(r, txt, train_texts) if txt else (False, "empty" if rec else "api_error")
        if ok:
            kept[r["id"]] = {**r, "sms": txt, "orig_sms": r["sms"], "split": f"{split}_rephrased",
                             "rephrased_by": REPHRASER, "rephrase_round": round_}
        return why
    for r in rows:
        why = try_accept(r, out1.get(r["id"]), 1)
        if why != "ok": failed.append(r)
    # second chance at a lower temperature for the ones that failed a check
    out2 = run_llm(f"rephrase_{split}_r2", failed, D.test_rephrase_messages, rephraser, REPHRASER_RPM, temperature=0.2, workers=8, **RP_EXTRA) if failed else {}
    for r in failed:
        reasons[try_accept(r, out2.get(r["id"]), 2)] += 1
    reasons["ok"] += len(rows) - len(failed)
    REPHRASED[split] = [kept[r["id"]] for r in rows if r["id"] in kept]
    COUNTS["rephrase"][split] = {"n": len(rows), "kept": len(REPHRASED[split]),
                                 "outcomes_after_retry": dict(reasons)}
    write_jsonl(f"{WORK}/data/{split}_rephrased.jsonl", REPHRASED[split])
    print(f"{split}: kept {len(REPHRASED[split])}/{len(rows)} | {dict(reasons.most_common())}")
    for x in REPHRASED[split][:2]:
        print(f"   original : {x['orig_sms']}\n   rephrased: {x['sms']}")
''')

md("## 9. Unrelated texts (should all be: not a transaction, every field empty)")
code(r'''
from datasets import load_dataset
UNREL, COUNTS["unrelated"] = [], {}
def add(source, lang, texts):
    for i, t in enumerate(texts):
        UNREL.append({"id": f"unrelated-{source}-{i:04d}", "family": f"unrelated_{source}", "lang": lang,
                      "template_id": source, "sms": t, "gold": dict(D.NULL_GOLD), "tags": ["unrelated"], "split": "unrelated"})
    COUNTS["unrelated"][source] = len(texts)

r = random.Random(SEED)
try:
    spam = load_dataset("ucirvine/sms_spam", split="train")
    ham_t = [x["sms"].strip() for x in spam if x["label"] == 0 and 20 <= len(x["sms"].strip()) <= 300]
    spam_t = [x["sms"].strip() for x in spam if x["label"] == 1 and 20 <= len(x["sms"].strip()) <= 300]
    add("sms_ham", "en", r.sample(ham_t, N_UNRELATED["sms_ham"]))
    add("sms_spam", "en", r.sample(spam_t, N_UNRELATED["sms_spam"]))
except Exception as e:
    print("UCI SMS Spam not loaded:", e); COUNTS["unrelated"]["sms_error"] = str(e)[:200]

def wiki(cfg, n, lang):
    sents = []
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
    for j, art in enumerate(ds):
        ss = D.sentences(art["text"])
        if ss: sents.append(r.choice(ss))
        if len(sents) >= n * 4 or j > 4000: break
    add(f"wiki_{lang}", lang, r.sample(sents, min(n, len(sents))))
for cfg, lang in [("20231101.en", "en"), ("20231101.hi", "hi")]:
    try: wiki(cfg, N_UNRELATED[f"wiki_{lang}"], lang)
    except Exception as e:
        print(f"Wikipedia {cfg} not loaded:", e); COUNTS["unrelated"][f"wiki_{lang}_error"] = str(e)[:200]
write_jsonl(f"{WORK}/data/unrelated.jsonl", UNREL)
print(COUNTS["unrelated"], "| total", len(UNREL))
for x in r.sample(UNREL, min(4, len(UNREL))): print("  ", x["family"], "|", x["sms"][:120])
''')

md("## 10. Filtering-count tables (Stage 1 and Stage 2 data)")
code(r'''
c2, c3 = V2_CURATION, COUNTS["v3"]
v2_rows = [
    ("Generated training SMS (generator, exact-duplicate guard across splits)", c2["rows_in"]),
    ("CuratorKIT exact de-dup (key = instruction + output)", c2["runs"][-1]["rows_exported"]),
    ("Our JSON gate (output must equal the gold label)", c2["runs"][-1]["rows_after_gate"]),
    ("Restored: all-null labels collapsed by the de-dup key, each re-checked by the gate", c2["rows_final"]),
    ("Grounding check (every labelled value appears in the SMS; by construction)", c2["rows_final"]),
]
v3_rows = [
    ("Stage 1 training SMS", len(V2["train"])),
    ("+ Aya Expanse rewrites requested", c3["rewrite_attempts"]),
    ("  rewrites passing hygiene, language, fact and meaning checks", c3["rewrites_accepted"]),
    ("Combined", c3["combined_in"]),
    ("CuratorKIT exact de-dup (SMS in instruction)", c3["after_curatorkit_exact_dedup"]),
    ("Our JSON gate", c3["after_json_gate"]),
    (f"CuratorKIT diversity gate (multilingual MiniLM, cos ≥ {DIVERSITY_THRESHOLD}, rewrites only)", c3["after_diversity_gate"]),
    ("Grounding check (hallucination)", c3["after_grounding_check"]),
]
def table(title, rows):
    s = f"**{title}**\n\n| Step | Rows |\n|---|---|\n" + "\n".join(f"| {a} | {b:,} |" for a, b in rows)
    return s
REASON_TXT = "\n".join(f"| {k} | {v} |" for k, v in sorted(c3["rewrite_reasons"].items(), key=lambda kv: -kv[1]))
md_out = (table("Stage 1 (v2) training data", v2_rows) + "\n\n" + table("Stage 2 (v3) training data", v3_rows)
          + "\n\n**Why rewrites were rejected**\n\n| Outcome | Rewrites |\n|---|---|\n" + REASON_TXT
          + "\n\n**Rephrased test sets** (" + REPHRASER + ")\n\n| Split | Kept / total |\n|---|---|\n"
          + "\n".join(f"| {k} | {v['kept']} / {v['n']} |" for k, v in COUNTS["rephrase"].items())
          + "\n\n**Unrelated texts**\n\n| Source | Rows |\n|---|---|\n"
          + "\n".join(f"| {k} | {v} |" for k, v in COUNTS["unrelated"].items() if not k.endswith("error")))
open(f"{WORK}/results/filter_tables.md", "w", encoding="utf-8").write(md_out)
COUNTS["models"] = {"train_rewriter": REWRITER, "test_rephraser": REPHRASER, "diversity_model": DIVERSITY_MODEL}
json.dump(COUNTS, open(f"{WORK}/results/filter_counts.json", "w"), indent=1, ensure_ascii=False, default=str)
print(md_out)
''')

md("## 11. Upload to Hugging Face")
code(r'''
CARD = f"""---
license: cc-by-4.0
language: [en, hi]
task_categories: [text-generation]
tags: [sms, upi, banking, information-extraction, structured-output, json, hindi, hinglish, india, finance, synthetic, curatorkit]
pretty_name: Tiny Aya SMS Extract, data v3
---
# Tiny Aya SMS Extract · data v3

Stage 2 data for fine-tuning Tiny Aya to turn Indian bank and UPI SMS (English, Hindi, Hinglish) into one 11-field JSON object.
Author: Anshul J Amin.

* `data/train_v3.jsonl`: Stage 1 training SMS plus fact-checked LLM rewrites ({REWRITER}, called through CuratorKIT); each rewrite keeps its source's gold label and `source_id`.
* `data/dev.jsonl`, `data/test_id.jsonl`, `data/test_ood.jsonl`, `data/challenge.jsonl`: unchanged Stage 1 evaluation sets.
* `data/*_rephrased.jsonl`: the test sets paraphrased by {REPHRASER} (a model not used for training data or as a compared system); same `id` as the original for paired comparison.
* `data/unrelated.jsonl`: texts that are not bank SMS (UCI SMS Spam Collection, CC BY 4.0; Wikipedia EN/HI, CC BY-SA 4.0). Gold: not a transaction, all fields null.
* `data/rewrites_log.jsonl`: every rewrite attempt with the check that accepted or rejected it.
* `results/filter_tables.md`, `results/filter_counts.json`: counts after every filtering step. `curatorkit_run/`: CuratorKIT manifest and provenance.

Bank names are fictional; no real customer SMS are included.

{md_out}
"""
open(f"{WORK}/README_v3.md", "w", encoding="utf-8").write(CARD)
if PUSH_TO_HUB and not QUICK_MODE:
    import shutil
    api = HfApi()
    stage = f"{WORK}/upload_v3"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(f"{stage}/data"); os.makedirs(f"{stage}/results")
    for f in ["train_v3.jsonl", "rewrites_log.jsonl", "unrelated.jsonl", "test_id_rephrased.jsonl",
              "test_ood_rephrased.jsonl", "challenge_rephrased.jsonl"]:
        if os.path.exists(f"{WORK}/data/{f}"):
            shutil.copy(f"{WORK}/data/{f}", f"{stage}/data/{f}")
    for name in ["dev", "test_id", "test_ood", "challenge"]:
        write_jsonl(f"{stage}/data/{name}.jsonl", V2[name])
    for f in ["filter_tables.md", "filter_counts.json"]:
        shutil.copy(f"{WORK}/results/{f}", f"{stage}/results/{f}")
    shutil.copytree(out_dir, f"{stage}/curatorkit_run", ignore=shutil.ignore_patterns("*alpaca*"))
    shutil.copy(f"{WORK}/README_v3.md", f"{stage}/README.md")
    api.create_repo(V3_DATA_REPO, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=stage, repo_id=V3_DATA_REPO, repo_type="dataset",
                      commit_message="Data v3: fact-checked rewrites, rephrased test sets, unrelated texts")
    try:
        api.add_collection_item(DATA_COLLECTION, item_id=V3_DATA_REPO, item_type="dataset", exists_ok=True)
    except Exception as e:
        print("collection:", e)
    print("uploaded: https://huggingface.co/datasets/" + V3_DATA_REPO)
else:
    print("not uploaded (QUICK_MODE or PUSH_TO_HUB=False). Set QUICK_MODE = False and run all.")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "none", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "stage2_data.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
