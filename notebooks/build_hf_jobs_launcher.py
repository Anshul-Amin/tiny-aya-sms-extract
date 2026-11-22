"""Build notebooks/hf_jobs_launcher.ipynb: a tiny CPU notebook that runs any project notebook
on a Hugging Face Jobs GPU, billed to the hackathon org. Used when Kaggle GPUs are queued.

The HF token stays a Kaggle secret: it is passed to the Job as a secret, never printed."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Run a project notebook on a Hugging Face Jobs GPU

**Tiny Aya SMS Extract** · Anshul J Amin · Lexsi AI Safety & Alignment Hackathon, Track 1

Kaggle GPUs can be queued for a long time. This CPU notebook starts the same notebook on a Hugging Face
Jobs GPU instead, billed to the `tinyaya-sms-extract` organisation (pay per minute). The executed notebook
is uploaded to the v3 data repo under `runs/` so its output can be read like a Kaggle log.

Settings: **no accelerator, Internet on**, Kaggle secret `HF_TOKEN` (needs permission to start Jobs for the org).""")

code(r'''
NOTEBOOK_URL = "https://raw.githubusercontent.com/Anshul-Amin/tiny-aya-sms-extract/main/notebooks/stage2_train_eval_full.ipynb"
RUN_NAME     = "stage2_train_eval_full"
FLAVOR       = "l4x1"            # 24 GB L4; t4-small is cheaper but slower
TIMEOUT      = "10h"
NAMESPACE    = "tinyaya-sms-extract"
IMAGE        = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
RESULTS_REPO = "Meowniece/tiny-aya-sms-extract-v3-data"
''')

code(r'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"], check=True)
from kaggle_secrets import UserSecretsClient
from huggingface_hub import run_job, inspect_job, fetch_job_logs
TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

SCRIPT = f"""
set -e
pip install -q nbconvert ipykernel huggingface_hub
python - <<'PY'
import urllib.request; urllib.request.urlretrieve("{NOTEBOOK_URL}", "nb.ipynb")
PY
set +e
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 nb.ipynb --output out.ipynb
STATUS=$?
python - <<'PY'
import os
from huggingface_hub import HfApi
for f in ["out.ipynb", "nb.ipynb"]:
    if os.path.exists(f):
        HfApi().upload_file(path_or_fileobj=f, path_in_repo="runs/{RUN_NAME}.ipynb", repo_id="{RESULTS_REPO}",
                            repo_type="dataset", commit_message="HF Jobs run: {RUN_NAME}")
        break
PY
exit $STATUS
"""
job = run_job(image=IMAGE, command=["bash", "-c", SCRIPT], flavor=FLAVOR, timeout=TIMEOUT,
              namespace=NAMESPACE, secrets={"HF_TOKEN": TOKEN}, token=TOKEN)
print("job id:", job.id)
print("url:", job.url)
''')

code(r'''
import time
for _ in range(20):          # watch the first ~10 minutes, then the job keeps going on its own
    st = inspect_job(job_id=job.id, namespace=NAMESPACE, token=TOKEN).status
    print(time.strftime("%H:%M:%S"), st.stage, st.message or "")
    if st.stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
        break
    time.sleep(30)
try:
    logs = list(fetch_job_logs(job_id=job.id, namespace=NAMESPACE, token=TOKEN))
    print("\n".join(logs[-40:]))
except Exception as e:
    print("logs not available yet:", e)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "none", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "hf_jobs_launcher.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
