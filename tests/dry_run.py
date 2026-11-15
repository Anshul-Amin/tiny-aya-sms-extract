"""CPU dry run of the notebook: real code for data, curation fallback, scoring, stats, charts,
error analysis, ROI and report rendering; a fake model replaces Tiny Aya / training / uploads."""
import json
import os
import random
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(ROOT, "notebooks", "sms_extraction_pipeline.ipynb")))
WORKDIR = "/tmp/p5_dry"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

# ---- fake torch ----
torch = types.ModuleType("torch")
torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
class _Cuda:
    is_available = staticmethod(lambda: False)
    is_bf16_supported = staticmethod(lambda: False)
    empty_cache = staticmethod(lambda: None)
    synchronize = staticmethod(lambda: None)
    get_device_name = staticmethod(lambda i=0: "FakeGPU")
torch.cuda = _Cuda()
torch.no_grad = lambda: (lambda f: f)
sys.modules["torch"] = torch
datasets = types.ModuleType("datasets")
class _DS:
    @staticmethod
    def from_list(x): return x
datasets.Dataset = _DS
sys.modules["datasets"] = datasets

MOCK_MODEL_CELL = r'''
import torch
BF16 = False; DTYPE = torch.float16
class _Tok:
    eos_token = "<eos>"; pad_token = "<pad>"; padding_side = "left"
    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": list(range(max(1, len(text) // 4)))}
tok = _Tok()
def to_prompt(messages):
    return "\n".join(m["content"] for m in messages)
def alpaca_prompt(sms):
    return f"### Instruction:\n{SYSTEM_PROMPT}\n\n### Input:\nSMS: {sms}\n\n### Response:\n"
_GOLD = {r["sms"]: r["gold"] for k in splits for r in splits[k]}
_QUALITY = {"base": 0.35, "ft": 0.85, "abl": 0.6}
def generate(prompts, mdl, batch=16, max_new=160):
    out = []
    rng = random.Random(len(prompts))
    for p in prompts:
        sms = p[p.rfind("SMS: ") + 5:].split("\n### Response")[0].strip()
        g = dict(_GOLD[sms])
        r = rng.random()
        q = _QUALITY.get(mdl, 0.5)
        if r < q:
            out.append(json.dumps(g, ensure_ascii=False))
        elif r < q + 0.1:
            out.append("Sure! Here is the data: " + str(g))   # invalid JSON
        else:
            g["amount"] = 999.0; g["balance"] = 1.0
            if not g["is_transaction"] and rng.random() < 0.5:
                g["is_transaction"] = True
            out.append("```json\n" + json.dumps(g, ensure_ascii=False) + "\n```")
    return out
model = "base"
'''
MOCK_FT_CELL = r'''
def free_gpu(): pass
def train_trl_fallback(rows, out_dir):
    return {"path": "trl_fallback", "format": "messages", "artifact": f"{out_dir}/final",
            "log_history": [{"step": i * 10, "loss": 2.0 / (i + 1)} for i in range(15)]}
def load_finetuned(info):
    return "abl" if "ablation" in info["artifact"] else "ft"
'''

ns = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = c["source"]
    if "def pip(" in src:
        src = "OK_ALIGN = OK_CURATOR = OK_AUDIT = False\ndef pip(*a): return False\nimport torch\n"
    elif "Hugging Face login" in src:
        src = "HF_USERNAME = 'tester'; token = None; api = None\n"
    elif "AutoModelForCausalLM" in src and "def load_base" in src:
        src = MOCK_MODEL_CELL
    elif "FT_INFO = finetune(train_rows" in src:
        # keep the REAL finetune()/save_train_log(); stub only the GPU training & loading
        src = src.replace('FT_INFO = finetune(train_rows, "ft_main")', MOCK_FT_CELL + '\nFT_INFO = finetune(train_rows, "ft_main")')
    if "QUICK_MODE  = False" in src:
        src = src.replace("QUICK_MODE  = False", "QUICK_MODE  = True").replace(
            "MAKE_GGUF   = True", "MAKE_GGUF   = False").replace("PUSH_TO_HUB = True", "PUSH_TO_HUB = False")
        src = src.replace('WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")',
                          f'WORK = "{WORKDIR}/work"')
    try:
        exec(compile(src, f"cell{i}", "exec"), ns)
    except Exception:
        print(f"\n!!! cell {i} failed:\n" + src[:300])
        raise
print("\nDRY RUN OK")
print(open(f"{WORKDIR}/work/ABSTRACT.md").read()[:1500])
