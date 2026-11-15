# Submission checklist (deadline: Nov 15, 11:59 PM AoE = **Mon Nov 16, 5:29 PM IST**)

## A. One-time setup (15 min)
- [ ] **Hugging Face:** sign up → open https://huggingface.co/CohereLabs/tiny-aya-global → accept the licence.
- [ ] HF → Settings → Access Tokens → **New token, type "Write"**. Copy it.
- [ ] **Kaggle:** sign up → verify your phone number (required for GPUs).
- [ ] **GitHub:** create a public repo, e.g. `tiny-aya-sms-extract`, and upload every file in this folder (drag and drop works).

## B. Run the pipeline on Kaggle
- [ ] Kaggle → *Create → New Notebook* → *File → Import Notebook* → upload `notebooks/sms_extraction_pipeline.ipynb`.
- [ ] Right panel: **Accelerator → GPU T4 x1**, **Internet → On**.
- [ ] *Add-ons → Secrets → Add* → name `HF_TOKEN`, value = your write token → tick it for this notebook.
- [ ] First cell: set `QUICK_MODE = True` → **Run All** (≈10 min). Everything should finish without red errors.
- [ ] Set `QUICK_MODE = False` → **Run All** (≈1.5–2.5 h). Keep the tab open, or use *Save Version → Save & Run All* so it runs in the background.
- [ ] Download from the *Output* panel: `ARTICLE.md`, `ABSTRACT.md`, `results/`, `figures/` (also uploaded to your HF dataset repo automatically).

**If something fails** (the notebook prints ⚠️ and falls back automatically in most cases):
| Symptom | What happens / what to do |
|---|---|
| CuratorKIT export can't be mapped | Falls back to exact de-dup; the report says so. Ask in Discord why, fix if time allows. |
| AlignTune errors or runs out of memory | Falls back to TRL + PEFT with identical settings; the report records `trl_fallback`. Mention it honestly. |
| AuditKIT `evaluate` errors | Metrics are still computed; `audit.auditkit_error` is saved in `metrics.json`. |
| Empty model outputs in fp16 | The notebook reloads in float32 automatically. |
| GGUF conversion fails | The adapter still works with transformers; run `space/app.py` locally for a live demo. |
| Gradio Space returns 402 | HF requires PRO for Gradio Spaces; the notebook publishes the free static results explorer instead. |
| Kaggle session dies mid-run | Re-run all. Training is ~20 min; each eval pass ~5–10 min. |

## C. Check what the notebook created on Hugging Face
- [ ] Model: `<you>/tiny-aya-sms-extract-lora` (+ `-gguf`, + `-lora-ablation-clean`)
- [ ] Dataset: `<you>/tiny-aya-sms-extract-data` (splits, predictions, metrics, figures, CuratorKIT run)
- [ ] Space: `<you>/tiny-aya-sms-extract-demo`. Open it and check the metrics table and the prediction browser load.
- [ ] Two collections: *"Tiny Aya SMS extraction: models"* and *"…: datasets"*. Make sure everything is **public**.
- [ ] **Logs are on HF** (the rules ask for models, adapters, logs and datasets): the model repo has `logs/` (training loss history, train info, loss curve); the dataset repo has `results/` (AuditKIT run records, metrics, every prediction, CuratorKIT report).
- [ ] Kaggle → *File → Download notebook* (the executed one, with outputs) → add it to your GitHub repo as `notebooks/executed_run.ipynb`. It is your full run log.

## D. Write-up (1–2 h)
- [ ] Open `ARTICLE.md`. Fill in section **5 "What we learned"** in your own words (the prompts tell you what to look at), add your GitHub link at the top, and add your name.
- [ ] Publish it as a Hugging Face article: https://huggingface.co/new-blog (or put it in the GitHub README if HF articles aren't available to you).
- [ ] Read `ABSTRACT.md` (must stay under 500 words) and adjust wording if a result surprised you.
- [ ] **Recommended:** open a Discord ticket in `#report_issues`, paste the abstract and article link, and ask for feedback before the deadline.

## E. Fill the submission form (hackathon.lexsi.ai → Submit)
| Field | What to put |
|---|---|
| Team email | the email you registered with |
| Abstract | contents of `ABSTRACT.md` |
| HF model collection | link to *"Tiny Aya SMS extraction: models"* |
| HF dataset collection | link to *"Tiny Aya SMS extraction: datasets"* |
| Write-up | your HF article link |
| GitHub repo | your repo link |
| Public demo | your Space link |
| Repos used | CuratorKIT, AlignTune, AuditKIT |
| Models used | CohereLabs/tiny-aya-global (core) |
| Track / problem statements | Track 1: #5 Structured output & extraction (primary), #2 Replacing a closed-model API, #4 Low-resource & multilingual |

- [ ] Use the **same Hugging Face username** as in your registration form (required for the HF credits).
- [ ] Submit. You can resubmit until the deadline; the latest one counts.

## F. If you are shortlisted (told by Nov 16)
Stage 2 runs until **Nov 22, 11:59 PM AoE**: submit a **video presentation or slides** and use the extra $200 HF credits. Good uses: a second seed or a Tiny Aya Fire comparison, a larger hand-written challenge set checked by a native speaker, and a 5-minute walkthrough of the demo. Ask me for help with the slides.
