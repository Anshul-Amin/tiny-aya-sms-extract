# Tiny Aya · Indian bank & UPI SMS → JSON (English · Hindi · Hinglish)

**Lexsi AI Safety & Alignment Hackathon 2026 · Track 1 (Fine-tuning & Data Curation)**
**Problem statements:** #5 Structured output & extraction (primary), #2 Replacing a closed-model API, #4 Low-resource & multilingual / code-mixed
**Model:** `CohereLabs/tiny-aya-global` (3.35B) + LoRA · **Lexsi repos used:** CuratorKIT → AlignTune → AuditKIT

Fine-tuning a small open model to read one transaction SMS and return one schema-valid JSON object, without inventing fields, and to recognise messages that mention money but are **not** transactions (OTPs, offers, reminders, future debits, money requests).

```
smsgen ──► CuratorKIT ──► AuditKIT (regex · zero-shot · few-shot) ──► AlignTune (LoRA SFT) ──► AuditKIT (after)
 exact      clean · dedup ·        same harness, same splits             TRL backend,           + ablation, stats,
 labels     provenance · gate                                            1× free T4             error analysis, ROI
```

## What's different here
1. **Unseen-format test split:** one template per (family, language) never appears in training.
2. **Hand-written challenge set:** 44 human-labelled, messy messages in all three languages.
3. **Hallucination is measured:** gold labels are grounded by construction (a field is labelled only if its text survived generation).
4. **Hard negatives + edge cases:** OTPs with amounts, credit-limit distractors, foreign currency, truncation, Devanagari digits, `Rs.450debited`, ALL CAPS.
5. **Ablation:** the same fine-tune without the robustness data.
6. **Fair baselines:** a careful regex parser, zero-shot and few-shot prompting of the same model.
7. **CPU deployment:** an 8-bit GGUF runs on CPU with llama.cpp; a free static Space lets anyone browse every test prediction against gold and the baselines.
8. **ROI the way the rules ask:** 10M SMS/month, measured SMS/s and tokens/s, GPU hours × $/hour, vs per-token API billing.
9. **Responsible data:** fully synthetic; bank names are fictional; no links or phone numbers.

## Repository layout
| Path | What |
|---|---|
| `notebooks/sms_extraction_pipeline.ipynb` | **The whole pipeline** (self-contained; run on Kaggle/Colab T4) |
| `src/smsgen.py` | Generator: 111 templates, 14 families × 3 languages, edge-case perturbations, held-out split |
| `src/scoring.py` | JSON extraction, schema validation, field metrics, hallucination/omission, McNemar, bootstrap |
| `src/regex_baseline.py` | Rule-based baseline |
| `data/challenge_set.py` → `.jsonl` | Hand-written challenge set |
| `space/` | Gradio live demo (GGUF on CPU, transformers fallback); run locally, or on a Space with HF PRO |
| `space_static/` | Static results explorer published as the free HF Space |
| `docs/ARTICLE_TEMPLATE.md`, `docs/ABSTRACT_TEMPLATE.md` | Filled in automatically by the notebook with real numbers |
| `tests/test_core.py` | Model-free tests (labels grounded in text, no leakage, scoring) |
| `tests/dry_run.py` | Runs the entire notebook on CPU with a fake model (plumbing test) |
| `notebooks/build_notebook.py` | Rebuilds the notebook from the source files |

## Reproduce
1. Hugging Face: accept the licence on the `CohereLabs/tiny-aya-global` model page; create a **write** token.
2. Kaggle → New Notebook → *File → Import Notebook* → `notebooks/sms_extraction_pipeline.ipynb`.
   Settings: **Accelerator GPU T4 x1**, **Internet on**. Add-ons → Secrets → `HF_TOKEN`.
3. Set `QUICK_MODE = True`, *Run all* (≈10 min smoke test). Then `QUICK_MODE = False`, *Run all* (≈1.5–2.5 h).
4. Outputs: `/kaggle/working/` → `results/` (metrics, predictions, AuditKIT records, training logs, ROI), `figures/`, `ARTICLE.md`, `ABSTRACT.md`, plus HF repos (adapter + `logs/`, GGUF, dataset), two collections and the results-explorer Space.

Local checks (no GPU): `python tests/test_core.py` and `python tests/dry_run.py`.

## Licences
Code: MIT. Data: CC-BY-4.0. The model adapter inherits Tiny Aya's **CC-BY-NC** licence (non-commercial). Lexsi toolkits are under their own source-available licences.
