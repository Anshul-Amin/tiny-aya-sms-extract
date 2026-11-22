"""Builds notebooks/stage2_train_eval.ipynb from the Stage 1 pipeline notebook.

Changes: Tiny Aya Fire, data v3 loaded from the Hub (no generation / curation here), extra evaluation splits
(rephrased test sets, unrelated texts), the Stage 1 adapters (Global v2, Fire v2) re-evaluated on every split,
robustness and unrelated-text tables, and paired comparisons between models.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(ROOT, "notebooks", "sms_extraction_pipeline.ipynb"), encoding="utf-8"))
cells = nb["cells"]


def find(sub, kind=None):
    hits = [i for i, c in enumerate(cells) if sub in c["source"] and (kind is None or c["cell_type"] == kind)]
    assert len(hits) == 1, (sub, hits)
    return hits[0]


def replace_in(sub, old, new, kind=None):
    i = find(sub, kind)
    assert old in cells[i]["source"], (sub, old[:60])
    cells[i]["source"] = cells[i]["source"].replace(old, new, 1)


def set_cell(sub, src, kind=None):
    cells[find(sub, kind)]["source"] = src.strip("\n")


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def mdc(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def wrap_try(sub):
    i = find(sub, "code")
    body = "\n".join("    " + l for l in cells[i]["source"].splitlines())
    cells[i]["source"] = "try:\n" + body + "\nexcept Exception:\n    traceback.print_exc()\n    print('⚠️ this optional step failed; continuing')"


# ---- title -------------------------------------------------------------------------------------
cells[0]["source"] = """# Stage 2 · Train Tiny Aya Fire on data v3 and evaluate everything

**Tiny Aya SMS Extract** · Anshul J Amin · Lexsi AI Safety & Alignment Hackathon, Track 1

* **Train:** LoRA fine-tune of `CohereLabs/tiny-aya-fire` with **AlignTune** on data v3 (Stage 1 training SMS + fact-checked Aya Expanse rewrites, curated with **CuratorKIT**). Same settings as Stage 1, so the only change from the Stage 1 Fire model is the data.
* **Evaluate with AuditKIT** on 7 sets: seen formats, unseen formats, hand-written, the three **rephrased** versions, and **unrelated texts**. Systems: regex, zero-shot, few-shot, the Stage 1 adapters (**Global v2**, **Fire v2**) and the new **Fire v3**.
* **Report:** main tables, original vs rephrased (paired, with 95% bootstrap intervals and McNemar tests), unrelated-text false-transaction rate, model-vs-model significance, throughput.

**How to run (Kaggle):** *File → Import Notebook* → this file. Settings: **GPU T4 x1**, **Internet ON**. Secret: `HF_TOKEN` (write token, with access to both gated Tiny Aya models). Run `QUICK_MODE = True` first (~20 min), then `QUICK_MODE = False` with *Save Version → Save & Run All* (~6 h)."""

# ---- config --------------------------------------------------------------------------------------
replace_in("CONFIG (edit me)", 'PROJECT     = "tiny-aya-sms-extract-v2"  # prefix for all HF repos (v2 = 3-epoch run; v1 repos are kept)',
           'PROJECT     = "tiny-aya-sms-extract-fire-v3"  # prefix for the new HF repos')
replace_in("CONFIG (edit me)", 'BASE_MODEL  = "CohereLabs/tiny-aya-global"', 'BASE_MODEL  = "CohereLabs/tiny-aya-fire"')
replace_in("CONFIG (edit me)", "RUN_ABLATION = True ", "RUN_ABLATION = False")
replace_in("CONFIG (edit me)", "MAKE_GGUF   = True ", "MAKE_GGUF   = False")
replace_in("CONFIG (edit me)", "QUICK_MODE  = False", "QUICK_MODE  = True ")
replace_in("CONFIG (edit me)", "N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD = 1600, 120, 300, 300   # sized to fit one free T4 session",
           'V3_DATA_REPO = "Meowniece/tiny-aya-sms-extract-v3-data"\n'
           'PRIOR_ADAPTERS = {   # Stage 1 adapters, re-evaluated on every split\n'
           '    "fire_v2":   ("CohereLabs/tiny-aya-fire",   "Meowniece/tiny-aya-sms-extract-fire-lora"),\n'
           '    "global_v2": ("CohereLabs/tiny-aya-global", "Meowniece/tiny-aya-sms-extract-v2-lora"),\n'
           '}\n'
           'N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD = None, None, None, None   # None = use every row of data v3')
replace_in("CONFIG (edit me)", "    N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD = 160, 30, 40, 40",
           "    N_TRAIN, N_DEV, N_TEST_ID, N_TEST_OOD = 120, 20, 16, 16")

# ---- section 3: load data v3 instead of generating ----------------------------------------------
cells[find("## 3. Generate the dataset", "markdown")]["source"] = (
    "## 3. Load data v3 from the Hub\nBuilt by `stage2_data.ipynb`: Stage 1 training SMS + fact-checked rewrites, the unchanged "
    "Stage 1 evaluation sets, their rephrased versions and the unrelated texts.")
set_cell("ds = build_dataset(N_TRAIN", r'''
from collections import Counter
from huggingface_hub import hf_hub_download

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_v3(name):
    p = hf_hub_download(V3_DATA_REPO, f"data/{name}.jsonl", repo_type="dataset")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

files = {"train": "train_v3", "dev": "dev", "test_id": "test_id", "test_ood": "test_ood", "challenge": "challenge",
         "test_id_rephrased": "test_id_rephrased", "test_ood_rephrased": "test_ood_rephrased",
         "challenge_rephrased": "challenge_rephrased", "unrelated": "unrelated"}
splits = {k: load_v3(v) for k, v in files.items()}
limits = {"train": N_TRAIN, "dev": N_DEV, "test_id": N_TEST_ID, "test_ood": N_TEST_OOD, "challenge": N_TEST_ID,
          "test_id_rephrased": N_TEST_ID, "test_ood_rephrased": N_TEST_OOD, "challenge_rephrased": N_TEST_ID,
          "unrelated": N_TEST_ID}
if QUICK_MODE:
    rq = random.Random(SEED)
    splits = {k: (rq.sample(v, min(limits[k], len(v))) if limits[k] else v) for k, v in splits.items()}
for r in splits["unrelated"]:
    r.setdefault("tags", ["unrelated"])

for k, rows in splits.items():
    write_jsonl(f"{WORK}/data/{k}.jsonl", rows)
write_jsonl(f"{WORK}/data/train_curated.jsonl", splits["train"])
write_jsonl(f"{WORK}/data/challenge_set.jsonl", splits["challenge"])

# leakage checks (rewrites carry their source template id + "+rw")
train_sms = {r["sms"] for r in splits["train"]}
for k in [k for k in splits if k != "train"]:
    assert not train_sms & {r["sms"] for r in splits[k]}, f"leak into {k}"
base_tpl = lambda t: t.replace("+rw", "")
assert not {base_tpl(r["template_id"]) for r in splits["test_ood"]} & {base_tpl(r["template_id"]) for r in splits["train"]}
print("no SMS overlap and no OOD-template overlap with train ✓")

summary = {k: {"n": len(v), "langs": dict(Counter(r["lang"] for r in v)),
               "non_txn": sum(not r["gold"]["is_transaction"] for r in v)} for k, v in splits.items()}
print(json.dumps(summary, indent=1, ensure_ascii=False))
print("rewrites in train:", sum("llm_rewrite" in r.get("tags", []) for r in splits["train"]))
''', "code")

# ---- section 4: curation already done in stage2_data.ipynb -------------------------------------
cells[find("## 4. CuratorKIT", "markdown")]["source"] = (
    "## 4. CuratorKIT (already run)\nData v3 was curated in `stage2_data.ipynb` (exact de-dup, JSON gate, diversity gate, "
    "grounding check). Its per-step counts are loaded here and saved with the results.")
set_cell("def run_curator(", r'''
train_rows = splits["train"]
try:
    curation = json.load(open(hf_hub_download(V3_DATA_REPO, "results/filter_counts.json", repo_type="dataset")))
except Exception as e:
    curation = {"note": f"filter_counts.json not found: {e}"}
curation["rows_final"] = len(train_rows)
# generation budget: longest gold answer (pessimistic ~2.2 chars/token) + margin; StopAfterJson ends early anyway
_longest = max(len(gold_to_str(r["gold"])) for k in splits for r in splits[k])
MAX_NEW_TOKENS = max(MAX_NEW_TOKENS, int(_longest / 2.2) + 48)
print("MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
json.dump(curation, open(f"{WORK}/results/curation_report.json", "w"), indent=1)
print(json.dumps(curation.get("v3", curation), indent=1, ensure_ascii=False, default=str)[:2500])
''', "code")

# ---- evaluation splits -----------------------------------------------------------------------------
replace_in('EVAL_SPLITS = ["test_id", "test_ood", "challenge"]', 'EVAL_SPLITS = ["test_id", "test_ood", "challenge"]',
           'EVAL_SPLITS = [s for s in ["test_id", "test_ood", "challenge", "test_id_rephrased", "test_ood_rephrased",\n'
           '                           "challenge_rephrased", "unrelated"] if splits.get(s)]')

# ---- after section 8: Stage 1 adapters on every split ---------------------------------------------
i8 = find("SYSTEMS[\"fine_tuned\"] = lambda xs", "code")
cells.insert(i8 + 1, mdc("## 8b. Stage 1 adapters on every split\nThe Stage 1 Fire (v2) and Global (v2) adapters are evaluated with the same prompts and scorer, so every comparison below is paired on the same messages."))
cells.insert(i8 + 2, code(r'''
from peft import PeftModel
V3_TOK = tok
def eval_prior(name, base_id, adapter_id):
    global model, tok
    free_gpu()
    tok = AutoTokenizer.from_pretrained(base_id); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=DTYPE, device_map=DEVICE_MAP)
    model = PeftModel.from_pretrained(base, adapter_id).eval()
    SYSTEMS[name] = lambda xs: generate([to_prompt(build_messages(x)) for x in xs], model)
    evaluate_system(name)
for name, (base_id, adapter_id) in PRIOR_ADAPTERS.items():
    try:
        eval_prior(name, base_id, adapter_id)
    except Exception:
        traceback.print_exc(); print(f"⚠️ {name} could not be evaluated; continuing")
free_gpu()
tok = V3_TOK
json.dump({f"{s}/{sp}": v["metrics"] for (s, sp), v in RESULTS.items()},
          open(f"{WORK}/results/metrics_checkpoint.json", "w"), indent=1, default=str)
'''))

# ---- section 10 tweaks -------------------------------------------------------------------------------
replace_in('systems = [s for s in ["regex", "zero_shot", "few_shot", "ft_no_robust_data", "fine_tuned"]',
           'systems = [s for s in ["regex", "zero_shot", "few_shot", "ft_no_robust_data", "fine_tuned"]',
           'systems = [s for s in ["regex", "zero_shot", "few_shot", "ft_no_robust_data", "global_v2", "fire_v2", "fine_tuned"]')
replace_in("def bar(metric, title, fname)", 'ax.set_xticklabels(["seen formats", "unseen formats", "hand-written"])',
           'ax.set_xticklabels([sp.replace("_rephrased", "\\nrephrased") for sp in EVAL_SPLITS], fontsize=7)')
replace_in("def bar(metric, title, fname)", 'fig, ax = plt.subplots(figsize=(9, 4))', 'fig, ax = plt.subplots(figsize=(12, 4))')

i10 = find("def bar(metric, title, fname)", "code")
cells.insert(i10 + 1, mdc("## 10b. Robustness, unrelated texts and model-vs-model comparisons"))
cells.insert(i10 + 2, code(r'''
def em_by_id(s, sp):
    return {r["id"]: bool(x["exact_match"]) for r, x in zip(splits[sp], RESULTS[(s, sp)]["scored"])}

# 1) original vs rephrased, paired on the same message ids
ROBUST_ROWS = []
rob_lines = ["| System | Set | n | Original EM [95% CI] | Rephrased EM [95% CI] | Δ (points) | McNemar p |",
             "|---|---|---|---|---|---|---|"]
for s in systems:
    for a in ["test_id", "test_ood", "challenge"]:
        b = f"{a}_rephrased"
        if (s, a) not in RESULTS or (s, b) not in RESULTS: continue
        ea, eb = em_by_id(s, a), em_by_id(s, b)
        ids = [i for i in eb if i in ea]
        va, vb = [ea[i] for i in ids], [eb[i] for i in ids]
        ca, cb = S.bootstrap_ci([float(x) for x in va], seed=SEED), S.bootstrap_ci([float(x) for x in vb], seed=SEED)
        mc = S.mcnemar_exact(va, vb)
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        ROBUST_ROWS.append({"system": s, "split": a, "n": len(ids), "orig": ma, "orig_ci": ca, "reph": mb, "reph_ci": cb, "p": mc["p_value"]})
        rob_lines.append(f"| {s} | {a} | {len(ids)} | {pct(ma)} [{pct(ca[0])}–{pct(ca[1])}] | {pct(mb)} [{pct(cb[0])}–{pct(cb[1])}] | {100 * (mb - ma):+.1f} | {mc['p_value']:.2g} |")
ROBUST_TABLE = "\n".join(rob_lines)
print(ROBUST_TABLE)

# 2) unrelated texts: should be "not a transaction" with every field empty
UNREL_TABLE = ""
if "unrelated" in EVAL_SPLITS:
    ul = ["| System | n | Says transaction ↓ | Fills any field ↓ | All fields correct (EM) | Schema valid |", "|---|---|---|---|---|---|"]
    for s in systems:
        if (s, "unrelated") not in RESULTS: continue
        sc = RESULTS[(s, "unrelated")]["scored"]
        n = len(sc)
        says = sum(x["pred_is_txn"] is True for x in sc) / n
        fills = sum((x.get("hallucinated") or 0) > 0 for x in sc) / n
        ul.append(f"| {s} | {n} | {pct(says)} | {pct(fills)} | {pct(sum(x['exact_match'] for x in sc) / n)} | {pct(sum(x['schema_valid'] for x in sc) / n)} |")
    UNREL_TABLE = "\n".join(ul)
    print("\n" + UNREL_TABLE)
    for s in [x for x in ["few_shot", "fire_v2", "fine_tuned"] if (x, "unrelated") in RESULTS]:
        fam = Counter(); tot = Counter()
        for r, x in zip(splits["unrelated"], RESULTS[(s, "unrelated")]["scored"]):
            tot[r["family"]] += 1; fam[r["family"]] += x["pred_is_txn"] is True
        print(f"  {s}: says-transaction by source:", {k: f"{fam[k]}/{tot[k]}" for k in tot})

# 3) model vs model (exact match, paired): Global v2 vs Fire v2 vs Fire v3
pair_lines = ["| A | B | Set | A EM | B EM | B-only correct | A-only correct | McNemar p |", "|---|---|---|---|---|---|---|---|"]
PAIRS = {}
for a_sys, b_sys in [("global_v2", "fire_v2"), ("fire_v2", "fine_tuned"), ("global_v2", "fine_tuned")]:
    for sp in EVAL_SPLITS:
        if (a_sys, sp) not in RESULTS or (b_sys, sp) not in RESULTS: continue
        ea, eb = em_by_id(a_sys, sp), em_by_id(b_sys, sp)
        ids = [i for i in ea if i in eb]
        mc = S.mcnemar_exact([ea[i] for i in ids], [eb[i] for i in ids])
        PAIRS[(a_sys, b_sys, sp)] = mc
        pair_lines.append(f"| {a_sys} | {b_sys} | {sp} | {pct(sum(ea[i] for i in ids) / len(ids))} | {pct(sum(eb[i] for i in ids) / len(ids))} | {mc['b_only']} | {mc['a_only']} | {mc['p_value']:.2g} |")
PAIR_TABLE = "\n".join(pair_lines)
print("\n" + PAIR_TABLE)

open(f"{WORK}/results/stage2_tables.md", "w", encoding="utf-8").write(
    "\n\n".join([f"### {sp} (n={len(splits[sp])})\n{TABLES[sp]}" for sp in EVAL_SPLITS])
    + "\n\n### Original vs rephrased (paired)\n" + ROBUST_TABLE
    + "\n\n### Unrelated texts\n" + UNREL_TABLE
    + "\n\n### Model vs model\n" + PAIR_TABLE
    + "\n\n### Fine-tuned (v3) vs every other system\n" + SIG_TABLE
    + "\n\n### Language breakdown (unseen formats)\n" + LANG_TABLES.get("test_ood", ""))
json.dump({"robustness": ROBUST_ROWS, "pairs": {"|".join(k): v for k, v in PAIRS.items()}},
          open(f"{WORK}/results/stage2_stats.json", "w"), indent=1, default=str)
print("saved results/stage2_tables.md")
'''))

# ---- model card title, optional steps guarded ---------------------------------------------------------
replace_in("MODEL_CARD = f", "# Tiny Aya Global, fine-tuned for Indian bank/UPI SMS → JSON",
           "# {BASE_MODEL.split('/')[-1]} (data v3), fine-tuned for Indian bank/UPI SMS → JSON")
for sub in ["SPACE_STATIC_HTML.replace", "def _coll(title, desc, items)"]:
    try:
        wrap_try(sub)
    except AssertionError:
        pass
i14 = find("## 14. Render the Hugging Face article", "markdown")
for j in range(i14, len(cells)):
    if cells[j]["cell_type"] == "code":
        cells[j]["source"] = "print('article rendering skipped in Stage 2: the paper is written separately')"
cells[i14]["source"] = "## 14. (Stage 1 article rendering: skipped)"

nb["metadata"]["kaggle"] = {"accelerator": "gpu", "isInternetEnabled": True}
out = os.path.join(ROOT, "notebooks", "stage2_train_eval.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
