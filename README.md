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

## Results (final run: 3 epochs, Kaggle T4)

Exact match = all 11 fields correct. Hallucination = a field filled when the SMS doesn't contain it.

| System | Seen formats (300) | **Unseen formats (300)** | **Hand-written (44)** | Hallucination, unseen ↓ |
|---|---|---|---|---|
| Regex parser | 31.0 | 46.7 | 25.0 | 4.2 |
| Zero-shot (same model) | 5.7 | 10.0 | 9.1 | 51.7 |
| Few-shot (same model) | 26.0 | 21.0 | 25.0 | 21.3 |
| Fine-tuned without robustness data (ablation) | 59.7 | 50.7 | 47.7 | 42.9 |
| **Fine-tuned (ours)** | **92.0** | **85.0** | **70.5** | **2.0** |

Valid JSON 99.7–100%; non-transaction F1 98.2% on unseen formats; every improvement is significant (paired McNemar). Full write-up: [`docs/ARTICLE.md`](docs/ARTICLE.md) · abstract: [`docs/ABSTRACT.md`](docs/ABSTRACT.md).

**Hugging Face:** [LoRA adapter](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-lora) · [GGUF](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-gguf) · [Ablation adapter](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-lora-ablation-clean) · [Dataset, predictions & logs](https://huggingface.co/datasets/Meowniece/tiny-aya-sms-extract-v2-data) · [Results explorer](https://huggingface.co/spaces/Meowniece/tiny-aya-sms-extract-v2-demo) · [Models collection](https://huggingface.co/collections/Meowniece/tiny-aya-sms-extraction-models-6af9a1a0ced10d8fb22b4ed6) · [Datasets collection](https://huggingface.co/collections/Meowniece/tiny-aya-sms-extraction-datasets-6afa2f2876dd2d18a69a3e77)

The `-v2` repos are the submission (3 epochs). The repos without `-v2` hold the earlier 1-epoch run, kept for transparency (exact match 68.7 / 76.7 / 59.1).

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
| `docs/ARTICLE.md`, `docs/ABSTRACT.md` | Final write-up and abstract (from the 3-epoch run) |
| `docs/ARTICLE_TEMPLATE.md`, `docs/ABSTRACT_TEMPLATE.md` | Templates the notebook fills with real numbers |
| `tests/test_core.py` | Model-free tests (labels grounded in text, no leakage, scoring) |
| `tests/dry_run.py` | Runs the entire notebook on CPU with a fake model (plumbing test) |
| `notebooks/build_notebook.py` | Rebuilds the notebook from the source files |

## Reproduce
1. Hugging Face: accept the licence on the `CohereLabs/tiny-aya-global` model page; create a **write** token.
2. Kaggle → New Notebook → *File → Import Notebook* → `notebooks/sms_extraction_pipeline.ipynb`.
   Settings: **Accelerator GPU T4 x1**, **Internet on**. Add-ons → Secrets → `HF_TOKEN`.
3. Set `QUICK_MODE = True`, *Run all* (≈10 min smoke test). Then `QUICK_MODE = False`, *Run all* (≈4.5 h on a T4).
4. Outputs: `/kaggle/working/` → `results/` (metrics, predictions, AuditKIT records, training logs, ROI), `figures/`, `ARTICLE.md`, `ABSTRACT.md`, plus HF repos (adapter + `logs/`, GGUF, dataset), two collections and the results-explorer Space.

Local checks (no GPU): `python tests/test_core.py` and `python tests/dry_run.py`.

## Licences
Code: MIT. Data: CC-BY-4.0. The model adapter inherits Tiny Aya's **CC-BY-NC** licence (non-commercial). Lexsi toolkits are under their own source-available licences.
