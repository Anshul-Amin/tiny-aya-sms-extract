# Turning Indian bank & UPI SMS into reliable JSON: a fine-tuned 3B open model that replaces a closed-LLM API (English, Hindi, Hinglish)

**One-line claim:** A LoRA fine-tune of Cohere's Tiny Aya Global (3.35B), trained on synthetic, CuratorKIT-curated data, turns Indian bank and UPI SMS into exact 11-field JSON, reaching **85.0%** exact match on message formats it never saw, against **21.0%** for few-shot prompting of the same model, with **2.0%** hallucinated fields against 21.3%.

**Author:** Anshul J Amin (solo entry; Hugging Face: [Meowniece](https://huggingface.co/Meowniece))

## Tracks and problem statements

- **Track 1: Fine-tuning & Data Curation**
  - **#5 Structured output & extraction** (primary)
  - **#2 Replacing a closed-model API**
  - **#4 Low-resource & multilingual / code-mixed** (Hindi in Devanagari and romanised Hinglish)

> ### For judges: one minute
> - **Task:** one bank/UPI SMS in (English, Hindi or Hinglish) → one JSON object with 11 fields, or `is_transaction: false` for OTPs, offers, reminders, future debits and balance alerts.
> - **Base model + method:** `CohereLabs/tiny-aya-global` (3.35B; also compared with `tiny-aya-fire`, which scored higher: 88.7% unseen, 79.5% hand-written) + LoRA (r=16) SFT with **AlignTune**, on 1,600 synthetic messages curated with **CuratorKIT**, evaluated with **AuditKIT**, on one free Kaggle T4.
> - **Before → after (same model, few-shot prompt → fine-tuned), exact match on all 11 fields:** seen formats **26.0% → 92.0%**, unseen formats **21.0% → 85.0%**, hand-written messages **25.0% → 70.5%**. Hallucinated fields on unseen formats **21.3% → 2.0%**.
> - **Why it's credible:** whole message formats held out of training, a separate hand-written test set, an ablation without the robustness data, paired McNemar tests (p ≤ 0.021 everywhere), bootstrap intervals, and every prediction published.
> - **Artifacts:** [models collection](https://huggingface.co/collections/Meowniece/tiny-aya-sms-extraction-models-6af9a1a0ced10d8fb22b4ed6) · [datasets collection](https://huggingface.co/collections/Meowniece/tiny-aya-sms-extraction-datasets-6afa2f2876dd2d18a69a3e77) · [results explorer](https://huggingface.co/spaces/Meowniece/tiny-aya-sms-extract-v2-demo) · [GitHub](https://github.com/Anshul-Amin/tiny-aya-sms-extract)

## Abstract

Indian finance apps read billions of bank and UPI transaction SMS a year, often by sending each message to a closed LLM API. We show that a small open model can do this job reliably, cheaply and on-premise, in English, Hindi (Devanagari) and code-mixed Hinglish.

We fine-tune Cohere Labs' Tiny Aya Global (3.35B) with LoRA, using a pipeline that chains three Lexsi toolkits: CuratorKIT (cleaning, de-duplication, provenance, plus a JSON quality gate), AlignTune (SFT with the TRL backend on one free T4 GPU) and AuditKIT (evaluation harness for every system and split). The model reads one SMS and returns one schema-valid JSON object with 11 fields: transaction flag, type, amount, currency, account, instrument, counterparty, date, reference, balance and status.

The training data is synthetic, then curated. A generator labels a field only if its text survives generation, which makes hallucination measurable, and every training row then passes through CuratorKIT (cleaning, de-duplication, provenance) and a JSON quality gate. It covers 14 message families, including five hard-negative families (OTPs, offers, bill reminders, future debits, balance alerts) that mention money but are not transactions. It also covers edge cases: Indian digit grouping, credit-limit distractors, foreign currency, truncation, Devanagari digits, missing spaces and casing noise. Bank names are fictional. To measure generalisation rather than memorisation, one template per family and language is held out entirely (test_ood). We also evaluate on a hand-written challenge set of realistic, messy messages.

On unseen formats the fine-tuned model reaches 85.0% exact match (all 11 fields correct), against 21.0% for few-shot prompting of the same model. On the hand-written set it reaches 70.5%, against 25.0% few-shot, 9.1% zero-shot and 25.0% for a hand-written regex parser. Its hallucination rate there is 10.1% (few-shot: 25.3%). An ablation trained on the same amount of clean data, without hard negatives or noise, reaches 47.7% exact match and 13.3% non-transaction F1 (full model: 70.5% and 81.5%), which isolates the effect of the robustness data. Paired McNemar tests and bootstrap intervals are reported.

At a realistic 10,000,000 SMS per month, self-hosting on a T4 costs about $871.52 (2,490 GPU-hours, from measured throughput), versus gpt-5-mini $2,198.59 and gpt-5.4-mini $5,464.01 at closed-API list prices. Because the model is small and open, extraction can run on-premise, so SMS need never leave the user's infrastructure; we also publish an 8-bit GGUF export for CPU inference and a free Space where anyone can browse every prediction. We release the adapter, the GGUF export, all data splits, every prediction, the CuratorKIT provenance and the AuditKIT run records, and we discuss limitations: synthetic training data, a small challenge set and a single training seed.

## Headline: before → after

| Same model, same test messages | Few-shot prompting (before) | Fine-tuned (after) |
|---|---|---|
| Exact match, seen formats (300) | 26.0% | **92.0%** |
| Exact match, **unseen formats** (300) | 21.0% | **85.0%** |
| Exact match, **hand-written** (44) | 25.0% | **70.5%** |
| Hallucinated fields, unseen formats ↓ | 21.3% | **2.0%** |
| Valid JSON, unseen formats | 88.3% | **99.7%** |

**Base model + method:** `CohereLabs/tiny-aya-global` (3.35B, fp16) + LoRA r=16 on the attention projections, supervised fine-tuning for 3 epochs with AlignTune (TRL backend) on 1,600 synthetic, CuratorKIT-curated SMS.

## 1. Problem statement and approach

### 1.1 Why this problem: the real use case

Expense trackers, lending apps and personal-finance tools in India read billions of transaction SMS a year, and many send each message to a closed LLM API to extract the transaction. That costs money per message, sends users' financial data to a third party, and is weakest exactly where Indian users are: Hindi script, Hinglish and Indian number formats.

A bank SMS looks simple until you meet real ones. *"Rs.2,999 spent at CROMA. Avl Lmt Rs.47,001"* has two amounts, and only one is the transaction; the other is a credit limit, not a balance. *"482913 is OTP for txn of INR 12,500 at AMAZON"* has an amount, a merchant and a card, but no money has moved. *"आपके खाते XX७७१२ में ₹२५,०००.०० जमा हुए"* uses Devanagari digits. Hinglish users type *"rs.500debit hue atm se"*.

For downstream finance apps, **a wrong value is worse than a missing one**: an invented balance or a phantom transaction corrupts a budget silently. So we optimise and report not just accuracy, but **hallucination**: how often the model fills a field that the SMS does not contain.

### 1.2 Approach

1. **Generate** a synthetic, exactly-labelled dataset of Indian bank/UPI SMS in three languages, with deliberate hard cases, and hold out whole message formats for testing (section 5).
2. **Curate** the training split with **CuratorKIT** plus our own JSON quality gate (section 5.4).
3. **Measure the starting point** with **AuditKIT**: a rule-based parser, and zero-shot and few-shot prompting of the same base model (section 7).
4. **Fine-tune** Tiny Aya Global with LoRA through **AlignTune** (section 7).
5. **Measure again** with the same harness on seen formats, unseen formats and a hand-written set, plus an ablation without the robustness data, significance tests and a cost comparison (section 8).

### 1.3 Research questions

- **RQ1:** Can a 3B open model, fine-tuned on synthetic data, extract all fields correctly on message formats it never saw, better than prompting the same model?
- **RQ2:** Does training on hard negatives, noise and incomplete messages reduce hallucinated fields and phantom transactions, compared with the same amount of clean data?
- **RQ3:** Can it replace a closed-model API at lower cost on free-tier hardware?

These were our questions from the start; we did not formally pre-register hypotheses.

### 1.4 Why this task, this model and this data

**Task.** SMS-to-JSON extraction is a high-volume, repetitive, well-defined job. That is exactly the kind of workload that is usually sent to a closed API and should not need one (Problem 2). It is checkable field by field, so improvement can be measured without an LLM judge (Problem 5). And in India it is inherently multilingual and code-mixed (Problem 4).

**Model: Tiny Aya Global (3.35B).**
- **Language balance.** It is the Tiny Aya variant tuned for balanced performance across languages and regions. Our inputs mix English, Hindi and romanised Hindi in the same message, so a balanced variant is the safer default. We also ran the identical pipeline on *Tiny Aya Fire* (South-Asia specialised); see section 8.4.
- **Size fits the budget.** At 3.35B parameters, LoRA fine-tuning and batched inference both fit on one free 16 GB T4, and we also publish an 8-bit GGUF export for CPU inference. Aya Expanse 8B/32B would be more capable, but they cannot be fine-tuned on free hardware or served cheaply on-premise, and cheap on-premise serving is the point of the ROI argument.
- **Small output space.** The output is short and structured (~80 tokens), so model size matters less than format discipline. Fine-tuning targets format discipline directly.

**Data: synthetic, then curated.** We generated the messages ourselves (exact labels, no personal data), then curated them with CuratorKIT and our own quality gate before training. Section 5 explains why, and exactly how.

## 2. Related work

- **No public benchmark exists** for Indian bank/UPI SMS extraction with field-level labels, and none covering Hindi or Hinglish; we release one (section 5).
- **Rule-based parsing:** [transaction-sms-parser](https://github.com/saurabhgupta050890/transaction-sms-parser) (MIT) is an open regex library for Indian transaction SMS; it ships one example message and reports no accuracy. Our own `regex` baseline plays this role in the comparison.
- **Small-model parsers:** [banking-sms-json-parser-v8](https://huggingface.co/rawsun00001/banking-sms-json-parser-v8) is a GPT-2 model trained on 60k synthetic English messages. Its data and evaluation are not released, it is English-only, and it does not report hallucination.
- **Method:** LoRA (Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685) via Hugging Face PEFT and TRL, through AlignTune.

## 3. Hackathon repos (kits) used, and how they connect

```
smsgen.py ──► CuratorKIT ──► JSON gate ──► AlignTune (LoRA SFT, TRL backend) ──► fine-tuned adapter
 (synthetic,    clean · exact dedup ·        1,600 rows                                   │
  exact labels)  provenance                                                                ▼
     │                                                                 AuditKIT: same harness for every system
     └──► dev / test_id / test_ood / challenge ───────────────────────► (regex · zero-shot · few-shot · ablation · fine-tuned)
                                                                        → metrics, McNemar, bootstrap, error analysis, ROI
```

### 3.1 CuratorKIT

- **Used for:** curating the training split before fine-tuning.
- **Features used:** `Curator(CuratorConfig(dataset=…, dedup="exact", clean=True/False, export_formats=["alpaca"]))`; exact de-duplication; text cleaning; provenance export (`manifest.json`, `checksums.txt`, `lexsi_provenance.json`, dataset card); MinHash near-duplicate detection (count reported, not applied).
- **What happened** (full counts in section 5.4): cleaning altered some messages so their labels no longer matched, so we kept the uncleaned run; exact de-duplication collapsed the 493 non-transaction rows into one because they share one all-null answer; our JSON gate restored every row whose answer is valid, giving 1,600 final rows. All CuratorKIT outputs are in the dataset repo under `curatorkit_run/`.

### 3.2 AlignTune

- **Used for:** the supervised fine-tuning of the main model and of the ablation model.
- **Features used:** `create_sft_trainer` with the **TRL backend**; built-in **PEFT LoRA** (`use_peft=True`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`); chat-format (`messages`) datasets, which AlignTune renders with the model's own chat template; `precision="fp16"`; gradient checkpointing; `max_seq_length`.
- **Our settings:** 3 epochs, per-device batch 2 × gradient accumulation 8 (effective batch 16), learning rate 2e-4, max length 832 tokens (measured from the data; see 8.6), seed 7.
- **AlignTune defaults we kept** (its documented defaults; the learning-rate curve in our logs matches them): AdamW (`adamw_torch`), cosine learning-rate schedule with 10% warm-up, weight decay 0.01; LoRA target modules chosen by AlignTune (`q_proj`, `k_proj`, `v_proj`, `o_proj`). The run used AlignTune throughout; the notebook's plain TRL + PEFT fallback was never needed.

### 3.3 AuditKIT

- **Used for:** running every system on every split through one evaluation harness, before and after fine-tuning.
- **Features used:** `auditkit.evaluate(samples, model=callable)` with `auditkit.Sample(input, target)`, producing run records with a headline score and a fingerprint for each (system, split); the run-comparison API (few-shot → fine-tuned) with per-sample flips.
- **What it reported:** with AuditKIT's own strict string matching, few-shot → fine-tuned exact match went 24.0% → 90.7% (seen formats, 204 examples went wrong → right, 4 right → wrong), 18.3% → 84.3% (unseen formats, 208 vs 10) and 25.0% → 70.5% (hand-written, 23 vs 3). Our field-aware scorer (section 6) gives slightly higher numbers because it accepts `1,23,456` vs `123456`-style formatting differences. Run records are in the dataset repo.

## 4. Models and licences

| Item | Details |
|---|---|
| **Base Aya model** | [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global), 3.35B parameters, loaded in fp16 (no quantization for training or evaluation), its own chat template, greedy decoding. Licence: **CC-BY-NC 4.0** (non-commercial). |
| **Our adapter** | LoRA r=16, α=32, dropout 0.05 on q/k/v/o projections ([`Meowniece/tiny-aya-sms-extract-v2-lora`](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-lora)). Inherits CC-BY-NC. |
| **Deployment export** | Merged model as an 8-bit **GGUF** (Q8_0) for llama.cpp ([`Meowniece/tiny-aya-sms-extract-v2-gguf`](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-gguf)); not benchmarked yet. |
| **Comparison base model** | [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (South-Asia variant), same pipeline and settings; adapter [`Meowniece/tiny-aya-sms-extract-fire-lora`](https://huggingface.co/Meowniece/tiny-aya-sms-extract-fire-lora). CC-BY-NC 4.0. |
| **Teacher model** | **None.** There is no distillation: labels come from the data generator and are exact by construction, so no model's mistakes are copied into the training data. |
| **Closed-model baseline** | Not run (no API budget). We compare cost only, using list prices (section 8.7); accuracy against a closed API is in "With more credits". |
| **Code / data** | Code MIT; data CC-BY-4.0. |

## 5. Datasets: synthetic, then curated

### 5.1 Why we generated the data instead of collecting it

- **No usable public dataset exists.** We found no public, field-labelled dataset of Indian bank/UPI SMS, and none in Hindi or Hinglish. The closest public projects are a regex library ([transaction-sms-parser](https://github.com/saurabhgupta050890/transaction-sms-parser), MIT) that ships a single example message, and a GPT-2 parser ([banking-sms-json-parser-v8](https://huggingface.co/rawsun00001/banking-sms-json-parser-v8)) trained on 60k synthetic English messages whose data is not released.
- **Real SMS are personal financial data.** They contain account digits, balances, names and UPI IDs. Collecting them from people's phones and publishing them would need informed consent under India's Digital Personal Data Protection Act, 2023, and could not be released as an open dataset. The hackathon also asks us to host the data publicly.
- **Exact labels make the safety metric possible.** Hand-labelling real SMS is slow and noisy. In particular, nobody can say with certainty whether an annotator "invented" a field. With generated data, every label is known, so we can measure hallucination exactly.
- **Control.** We can create the hard cases on purpose, such as OTPs that mention amounts or credit limits next to the real amount, and hold out entire message formats to test generalisation.

The risk of synthetic data is that a model learns the templates instead of the task. Sections 5.3 and 8 show how we test for that: a split made only of unseen formats, and a hand-written set from outside the generator.

### 5.2 How each message is built

The generator is `src/smsgen.py`; everything is seeded (seed 7), so the dataset is exactly reproducible.

1. **Templates.** We wrote **111 templates**: 9 transaction families × 3 languages × 3 templates, plus 5 non-transaction families × 3 languages × 2 templates. Families: UPI debit, UPI credit, card spend, ATM withdrawal, transfer in, transfer out (NEFT/IMPS/RTGS), refund/reversal, failed transaction, AutoPay/EMI/NACH, and the non-transactions OTP, promotional offer, bill due, scheduled (future) debit, and balance information. Templates follow the general shape of common Indian bank alerts but are written from scratch; no real message was copied. All bank names are fictional (e.g. Kestrel Bank, Monsoon Bank) and messages contain no links or phone numbers, so they can't be mistaken for, or reused as, real bank messages.
2. **Values.** Each slot is sampled independently:
   - **Amounts** log-uniform from ₹1 to ₹5,00,000 (60% with paise), written in Indian (1,23,456), Western (123,456) or plain grouping, with ₹ / Rs. / INR / रु. / रुपये / "rupaye". Card spends are sometimes in USD, EUR, GBP or AED.
   - **Dates** uniform over 2025–2026, in 8 day-first formats (14-11-26, 14/11/2026, 14Nov26, 2026-11-14, …).
   - **Accounts and cards** masked in 8 styles (XX1234, \*1234, ending 1234, XXXX-1234, …).
   - **References**: 12-digit UPI RRNs, 16-character NEFT/RTGS UTRs and alphanumeric references.
   - **Counterparties** from 23 merchants, 15 person names, 6 employers, subscriptions, billers and generated UPI IDs (e.g. `ramesh.k82@saffron`).
3. **Labels by construction.** Each template is split into segments. The first segment is always kept; each later segment is dropped with probability 0.15, and 10% of messages are truncated. **A field is labelled only if the segment that prints it survived**, so a gold label never asks for information the SMS does not contain. Non-transaction messages have every field null except `is_transaction: false`, even when they mention amounts, merchants or cards.
4. **Real-world noise.** 35% of messages get one or two perturbations that never change the label: missing or doubled spaces (`Rs.450debited`), bank headers (`BNYN:`) and footers, promotional tails with extra amounts ("Win vouchers worth Rs 500"), ALL CAPS or lowercase (English/Hinglish), and Devanagari digits (Hindi).
5. **Mix.** 30% of messages come from the non-transaction families (hard negatives).

Example (Hindi, with Indian grouping and a credit-limit distractor):

```
SMS:  कार्ड **** 6735से ₹1,64,945.77 का लेनदेन ZOMATO पर, 2025-02-07। संदर्भ 186747152035। उपलब्ध क्रेडिट लिमिट रुपये 64,582।
JSON: {"is_transaction": true, "type": "debit", "amount": 164945.77, "currency": "INR", "account_last4": "6735",
       "instrument": "CARD", "counterparty": "ZOMATO", "date": "2025-02-07", "reference": "186747152035",
       "balance": null, "status": "success"}
```

### 5.3 Splits that test generalisation, not memorisation

- **Template-level hold-out.** For every (family, language) pair, the last template is never used for training. `test_ood` contains **only** those 42 unseen formats.
- **No overlap.** No SMS appears twice in the generated data, and none of the hand-written challenge messages appears in it; a test checks both.
- **Hand-written challenge set (44 SMS).** Written and labelled by hand in styles the generator never produces: reordered fields, pipe-separated alerts, "Cr/Dr" abbreviations, money requests, cashback credits and mixed scripts.
- **Ablation data.** Same generator, same size, different seed, but with no hard negatives, no noise and no dropped fields.

| Split | Size | en / hi / hinglish | Non-transactions | Promo tails | Devanagari digits | Missing fields / truncated |
|---|---|---|---|---|---|---|
| train | 1,600 | 550 / 560 / 490 | 493 | 126 | 49 | 390 / 103 |
| dev (format choice only) | 120 | 48 / 33 / 39 | 38 | 11 | 2 | 37 / 6 |
| test_id (seen formats) | 300 | 94 / 114 / 92 | 84 | 34 | 12 | 90 / 22 |
| test_ood (**unseen formats**) | 300 | 98 / 109 / 93 | 85 | 19 | 12 | 68 / 16 |
| challenge (**hand-written**) | 44 | 19 / 12 / 13 | 14 | – | – | – |

### 5.4 Curation with CuratorKIT, and validation

Generation gives us raw data; curation decides what the model actually trains on. Every training row passed through the same curation pipeline, and every step is recorded in the dataset repo (`curatorkit_run/`).


- **CuratorKIT.** The training split is exported in Alpaca format and run through CuratorKIT for cleaning, exact de-duplication and provenance (manifest, checksums, `lexsi_provenance.json`, dataset card). Curation report: 1,600 rows in → CuratorKIT exact de-dup exported 1,108 (its exact matching ignores the SMS text, so the 493 non-transaction rows, which all share the same all-null answer, collapsed into one) → our JSON gate restored the 492 valid rows → 1,600 final; MinHash would have kept only 25.
- **Our JSON quality gate.** After curation, every row's output must still parse to JSON identical to its gold label, or it is dropped.
- **Why no near-duplicate removal.** We ran MinHash near-duplicate detection and report its count, but did not apply it: templated SMS are near-duplicates *by design* yet carry different labels, so removing them would delete exactly the minimal pairs the model needs.
- **Automated checks** (`tests/test_core.py`):
  - every gold value is grounded in the SMS text;
  - all 2,364 gold labels are schema-valid and score 100% against themselves (checked during the audit);
  - no SMS appears in more than one split;
  - the challenge set is valid.

## 6. Metrics, and why these

Higher is better (↑) unless marked lower is better (↓).


| Metric | Definition | Why it matters for this use case |
|---|---|---|
| **Exact match ↑** | all 11 fields correct | An app consumes the whole record; one wrong field (e.g. debit vs credit) breaks the budget. This is the headline metric. |
| **Field accuracy ↑** | per-field correctness on transaction SMS | Shows *where* errors are: amount vs counterparty vs date. |
| **Hallucination rate ↓** | non-null prediction where gold is null | In finance a made-up balance or reference is worse than a blank; this is the reliability metric. |
| **Omission rate ↓** | null prediction where gold has a value | The other side of the trade-off: a model that always says null would have zero hallucination. |
| **Non-transaction F1 ↑** | catching OTPs, offers, reminders, future debits | A phantom transaction is the most damaging error for downstream apps. |
| **Schema validity ↑** | parseable JSON, right keys and types | The integration contract: invalid output needs retries or a human. |

Matching is lenient on formatting (case, whitespace, `1,23,456` vs `123456`, Devanagari digits) and strict on meaning. Significance uses a paired exact McNemar test; 95% intervals use bootstrap resampling of test examples.

**What should not get worse:** we did not run a general-capability benchmark. The model is a task-specific adapter, which can be unloaded to recover the base model, but we have not measured how the merged model performs on other tasks (see Limitations).

## 7. Method and experiments

### 7.1 Systems compared
| System | What it is |
|---|---|
| `regex` | A careful hand-written rule parser (English + Hindi/Hinglish keywords, Indian grouping, day-first dates). A strong non-ML baseline, not a straw man. |
| `zero_shot` | Tiny Aya Global with the schema instructions only |
| `few_shot` | Same + 3 fixed exemplars from train (EN, HI, Hinglish negative) |
| `ft_no_robust_data` | LoRA fine-tune on the same number of **clean** examples: no hard negatives, no noise, no missing fields (ablation) |
| `fine_tuned` | **Ours:** LoRA fine-tune on the curated robustness data |

### 7.2 Training setup and hyper-parameters

- **Data:** 1,600 curated training messages (section 5); chat format with Tiny Aya's own template, system prompt = the schema instructions.
- **AlignTune (TRL backend):** LoRA r=16, α=32, dropout 0.05, targets q/k/v/o; lr 2e-4, cosine schedule, 10% warm-up, AdamW, weight decay 0.01; 3 epochs; batch 2 × accumulation 8 = 16 (300 optimizer steps); max length 832 tokens; fp16; gradient checkpointing; **96 min** on one T4.
- **Ablation (`ft_no_robust_data`):** identical settings and size, but trained on clean generated data with no hard negatives, no noise and no dropped fields (generator seed 8); 98 min.
- **Inference:** greedy decoding, batched, generation stops after the first complete JSON object; max new tokens sized from the longest gold answer. The prompt format (`chat` vs Alpaca) was chosen on dev only (chat: 95.8% vs 60.0%).
- **Seeds:** generator and training seed 7 (ablation data: seed 8); single training seed.

### 7.3 Experiments

1. Baselines: regex, zero-shot, few-shot (3 fixed exemplars from train).
2. Main fine-tune (3 epochs) and the ablation (3 epochs, clean data).
3. An earlier 1-epoch run, identical except for the number of epochs, kept and published as a negative result (section 8.6).
4. The same pipeline with **Tiny Aya Fire** as the base model, without the ablation and GGUF export (section 8.4).

## 8. Results

### 8.1 Same model before vs after, and all systems

#### Seen formats (test_id)
| System | Exact match | Field acc. | Halluc. ↓ | Omission ↓ | Non-txn F1 | Schema valid |
|---|---|---|---|---|---|---|
| regex | 31.0 | 75.6 | 12.9 | 27.6 | 67.0 | 100.0 |
| zero_shot | 5.7 | 79.5 | 65.4 | 3.1 | 60.2 | 86.0 |
| few_shot | 26.0 | 84.0 | 25.3 | 14.5 | 67.9 | 85.7 |
| ft_no_robust_data | 59.7 | 97.7 | 49.8 | 0.2 | 6.9 | 96.3 |
| **fine_tuned** | 92.0 | 98.7 | 0.2 | 0.3 | 100.0 | 100.0 |

#### Unseen formats (test_ood)
| System | Exact match | Field acc. | Halluc. ↓ | Omission ↓ | Non-txn F1 | Schema valid |
|---|---|---|---|---|---|---|
| regex | 46.7 | 91.3 | 4.2 | 9.9 | 92.7 | 100.0 |
| zero_shot | 10.0 | 82.3 | 51.7 | 1.9 | 81.9 | 82.0 |
| few_shot | 21.0 | 87.3 | 21.3 | 11.7 | 83.2 | 88.3 |
| ft_no_robust_data | 50.7 | 96.8 | 42.9 | 0.7 | 46.8 | 94.0 |
| **fine_tuned** | 85.0 | 97.8 | 2.0 | 0.2 | 98.2 | 99.7 |

#### Hand-written challenge set
| System | Exact match | Field acc. | Halluc. ↓ | Omission ↓ | Non-txn F1 | Schema valid |
|---|---|---|---|---|---|---|
| regex | 25.0 | 87.0 | 24.2 | 10.7 | 57.1 | 100.0 |
| zero_shot | 9.1 | 78.7 | 63.1 | 2.5 | 52.6 | 70.5 |
| few_shot | 25.0 | 84.7 | 25.3 | 15.3 | 80.0 | 86.4 |
| ft_no_robust_data | 47.7 | 96.7 | 46.5 | 0.4 | 13.3 | 97.7 |
| **fine_tuned** | 70.5 | 92.0 | 10.1 | 6.6 | 81.5 | 100.0 |

#### How precise are these numbers? (exact match, 95% bootstrap interval)
| System | Seen formats | Unseen formats | Hand-written |
|---|---|---|---|
| regex | 31.0 [26.0, 36.3] | 46.7 [41.0, 52.3] | 25.0 [13.6, 38.6] |
| few_shot | 26.0 [21.0, 31.0] | 21.0 [16.7, 25.7] | 25.0 [13.6, 38.6] |
| **fine_tuned** | **92.0 [88.7, 95.0]** | **85.0 [81.0, 88.7]** | **70.5 [56.8, 84.1]** |

The hand-written set is small (44 messages), so its interval is wide; the intervals never overlap with the baselines.

#### Is the improvement real? (paired exact McNemar test on exact match)
| Comparison | Split | FT wins | Other wins | McNemar p |
|---|---|---|---|---|
| fine_tuned vs regex | test_id | 183 | 0 | 1.6e-55 |
| fine_tuned vs zero_shot | test_id | 260 | 1 | 1.4e-76 |
| fine_tuned vs few_shot | test_id | 202 | 4 | 1.4e-54 |
| fine_tuned vs ft_no_robust_data | test_id | 104 | 7 | 2.8e-23 |
| fine_tuned vs regex | test_ood | 124 | 9 | 5.4e-27 |
| fine_tuned vs zero_shot | test_ood | 232 | 7 | 1.9e-59 |
| fine_tuned vs few_shot | test_ood | 204 | 12 | 3.2e-46 |
| fine_tuned vs ft_no_robust_data | test_ood | 113 | 10 | 3.1e-23 |
| fine_tuned vs regex | challenge | 23 | 3 | 8.8e-05 |
| fine_tuned vs zero_shot | challenge | 29 | 2 | 4.6e-07 |
| fine_tuned vs few_shot | challenge | 23 | 3 | 8.8e-05 |
| fine_tuned vs ft_no_robust_data | challenge | 13 | 3 | 0.021 |

#### By language: exact match on unseen formats
| System | en EM | hi EM | hinglish EM |
|---|---|---|---|
| regex | 63.3 | 36.7 | 40.9 |
| zero_shot | 11.2 | 11.0 | 7.5 |
| few_shot | 21.4 | 10.1 | 33.3 |
| ft_no_robust_data | 51.0 | 55.0 | 45.2 |
| **fine_tuned** | **83.7** | **90.8** | **79.6** |

#### By language: exact match on the challenge set
| System | en EM (n=19) | hi EM (n=12) | hinglish EM (n=13) |
|---|---|---|---|
| regex | 26.3 | 25.0 | 23.1 |
| zero_shot | 15.8 | 0.0 | 7.7 |
| few_shot | 31.6 | 16.7 | 23.1 |
| ft_no_robust_data | 36.8 | 50.0 | 61.5 |
| **fine_tuned** | **57.9** | **75.0** | **84.6** |

#### Edge cases: exact match on the challenge set
| Edge case (n) | regex | few_shot | ft_no_robust_data | fine_tuned |
|---|---|---|---|---|
| hard_negative (14) | 42.9 | 21.4 | 0.0 | 78.6 |
| reordered (3) | 0.0 | 33.3 | 66.7 | 100.0 |
| failed (3) | 0.0 | 100.0 | 100.0 | 33.3 |
| truncated (3) | 33.3 | 0.0 | 100.0 | 100.0 |
| otp_with_amount (3) | 100.0 | 0.0 | 0.0 | 66.7 |
| future_debit (3) | 66.7 | 0.0 | 0.0 | 100.0 |
| lakh_grouping (2) | 0.0 | 0.0 | 0.0 | 50.0 |
| limit_distractor (2) | 0.0 | 0.0 | 50.0 | 50.0 |
| foreign_currency (2) | 50.0 | 0.0 | 100.0 | 50.0 |
| reversal (2) | 0.0 | 0.0 | 50.0 | 50.0 |
| missing_space (2) | 50.0 | 50.0 | 100.0 | 100.0 |
| atm (2) | 50.0 | 50.0 | 50.0 | 50.0 |
| bill_due (2) | 0.0 | 50.0 | 0.0 | 100.0 |
| balance_only (2) | 50.0 | 0.0 | 0.0 | 100.0 |

*Each edge case has only 2–14 examples, so read these as qualitative.*

![Exact match](https://huggingface.co/datasets/Meowniece/tiny-aya-sms-extract-v2-data/resolve/main/figures/exact_match.png)
![Hallucination](https://huggingface.co/datasets/Meowniece/tiny-aya-sms-extract-v2-data/resolve/main/figures/hallucination.png)
![Non-transaction F1](https://huggingface.co/datasets/Meowniece/tiny-aya-sms-extract-v2-data/resolve/main/figures/nontxn_f1.png)

### 8.2 Sample inputs and outputs (hand-written set)

**Hindi, pipe-separated alert:** `बनयान बैंक: UPI रेफ़ 300198765432 | ₹1,299.00 खाता XX4321 से डेबिट | प्राप्तकर्ता MYNTRA | 12.11.2026`
- Few-shot: `"account_last4": "XX4321"`, `"status": null` (two fields wrong)
- Fine-tuned: `{"is_transaction": true, "type": "debit", "amount": 1299.0, "currency": "INR", "account_last4": "4321", "instrument": "UPI", "counterparty": "MYNTRA", "date": "2026-11-12", "reference": "300198765432", "balance": null, "status": "success"}` ✓

**OTP with an amount (not a transaction):** `Do not share. 482913 is OTP for txn of INR 12,500.00 at AMAZON on card XX4410. -Kestrel Bank`
- Few-shot: `is_transaction: false`, but still fills `amount: 12500`, `counterparty: "AMAZON"`, `account_last4: "4410"` (hallucinated fields)
- Fine-tuned: `is_transaction: false`, every other field null ✓

**Future auto-debit (not a transaction yet):** `Rs 499 will be auto-debited on 20-11-26 for your NETFLIX mandate from a/c XX1190. Ensure sufficient balance.`
- Few-shot: a completed debit of ₹499 to NETFLIX ✗
- Fine-tuned: `is_transaction: false` ✓

**Reversal:** `Rs 1,250 debited from your a/c XX3390 on 05-09-26 for UPI txn to IRCTC has been reversed. UPI Ref 512309871234.`
- Few-shot: `type: "debit"`, `account_last4: "XX3390"` ✗
- Fine-tuned: `type: "credit"`, `status: "reversed"`, `account_last4: "3390"` ✓

### 8.3 Findings

- **The model learned the task, not the templates.** On message formats it never saw, exact match is 85.0% versus 92.0% on seen formats: a 7-point gap. Hindi is the strongest language on unseen formats (90.8%).
- **Prompting alone is not reliable enough to ship.** Zero-shot output was valid JSON only 82.0% of the time on unseen formats and hallucinated 52% of null fields; few-shot still hallucinated 21%. The fine-tuned model is valid 99.7% of the time and hallucinates 2.0%.
- **The robustness data is what makes it safe.** Trained on the same amount of clean data, the ablation scores well on field accuracy but cannot tell an OTP, offer or future debit from a real transaction (non-transaction F1 46.8% vs 98.2% on unseen formats) and hallucinates 43% of null fields. The improvement is significant on every split (McNemar p ≤ 0.021).
- **Training long enough mattered as much as the data.** With one epoch the model reproduced only 69% of its own training answers (dev exact match 70%); three epochs reached 95.8% on dev, 92.0 / 85.0 / 70.5% on the test sets. Both runs are published.
- **Check what the trainer actually sees.** Tiny Aya's chat template adds a ~400-token default preamble, so our first run (max length 512) silently cut every JSON answer off: 99.8% token accuracy, ~2% exact match. A sanity check (can the model reproduce 16 training answers?) and a log of the last tokens of a training example now guard against this.
- **Still hard:** the direction of money in wordings the model never saw ("Salary aa gayi", "refund … processed to Card", "वेतन जमा" are credits, but are sometimes labelled debit), copying long digit strings exactly (a transposed account number, an amount read from Devanagari digits), and English messages in the hand-written set (57.9% vs 75–85% for Hindi/Hinglish).

Most frequent remaining errors (fine-tuned, unseen + challenge): type (31), amount (17), counterparty (12), reference (9), currency (9), is_transaction (8), across 58 wrong examples. Every error, with gold and prediction, is in `results/error_analysis_fine_tuned.csv`.

### 8.4 Tiny Aya Global vs Tiny Aya Fire

We re-ran the identical pipeline (same data, splits, prompts, LoRA settings, 3 epochs, seed) with **Tiny Aya Fire**, the South-Asia-specialised Tiny Aya model. Only the base model changed. To fit the deadline this run skipped the ablation and the GGUF export.

| Exact match | Seen formats | Unseen formats | Hand-written |
|---|---|---|---|
| Few-shot, Global | 26.0 | 21.0 | 25.0 |
| Few-shot, Fire | 34.3 | 30.3 | 29.5 |
| **Fine-tuned, Global** | 92.0 | 85.0 | 70.5 |
| **Fine-tuned, Fire** | **92.7** | **88.7** | **79.5** |

| Fine-tuned model | Hallucination (seen / unseen / hand-written) ↓ | Non-txn F1 (unseen) | Unseen-format EM: en / hi / hinglish | Dev EM | Training answers reproduced |
|---|---|---|---|---|---|
| Global | 0.2 / 2.0 / 10.1 | 98.2 | 83.7 / 90.8 / 79.6 | 95.8 | 94% |
| Fire | 0.3 / 3.1 / 6.6 | 97.0 | 85.7 / 93.6 / 86.0 | 96.7 | 100% |

- **Fire is better at both stages.** It is a stronger starting point (few-shot +5 to +9 points) and ends higher after fine-tuning, most clearly on the hand-written set (+9 points) and on Hinglish (+6.4 points on unseen formats). That fits its South-Asia specialisation.
- **The gains are modest and partly uncertain.** On the hand-written set, +9 points is 4 messages out of 44, and we did not test the Global-vs-Fire difference for significance. Hallucination is slightly higher for Fire on synthetic test sets and lower on the hand-written set.
- **Both fine-tunes beat every baseline by a wide margin** (McNemar p < 1e-6 for Fire against regex, zero-shot and few-shot on every split), so the method works with either base model.
- **Why Global remains the main model in this write-up:** it was chosen before this comparison and has the complete evidence (ablation, GGUF export). The Fire adapter, predictions and logs are published alongside it ([adapter](https://huggingface.co/Meowniece/tiny-aya-sms-extract-fire-lora), [data and predictions](https://huggingface.co/datasets/Meowniece/tiny-aya-sms-extract-fire-data), [results explorer](https://huggingface.co/spaces/Meowniece/tiny-aya-sms-extract-fire-demo)).

### 8.5 What this means in practice

- **Drop-in for an API call:** same input (one SMS), same output (one JSON object); the schema-validity rate shows how often a retry would be needed.
- **Safer by default:** a lower hallucination rate and higher non-transaction F1 mean fewer phantom transactions and invented balances in users' budgets.
- **Works for how Indians actually text:** Hindi script, Hinglish, Devanagari digits and Indian number grouping are supported natively, with no transliteration step.
- **Private:** a 3B model can run on-premise, so no financial SMS has to leave the company's infrastructure.

### 8.6 What didn't work (negative results)

- **First training run: answers cut off.** Tiny Aya's chat template adds a ~400-token default preamble, so training examples are ~754 tokens. With a 512-token limit, every JSON answer was truncated away: 99.8% training token accuracy but ~2% exact match. Fixed by measuring the length from the data; a sanity check now confirms the model reproduces its own training answers.
- **One epoch underfit.** The 1-epoch model reproduced only 69% of its own training answers and scored 68.7 / 76.7 / 59.1% exact match. Three epochs gave 92.0 / 85.0 / 70.5%. Both runs are published.
- **CuratorKIT exact de-duplication** collapsed all non-transaction examples into one (they share an all-null answer), and **text cleaning** changed some messages so their labels no longer matched. We kept the uncleaned run and restored valid rows with our JSON gate.
- **A wrong precision value** (`"float16"` instead of `"fp16"`) was silently ignored by AlignTune, which fell back to emulated bf16 on the T4 and ran about 6× slower; fixed after an audit of the pipeline.
- **Free Colab** ran out of memory during training twice; all final runs used Kaggle.
- **A live Gradio demo** needs a paid Hugging Face plan, so the public Space is a static results explorer; the live app is in the GitHub repo.

### 8.7 Cost and compute

Scenario: **10,000,000 SMS / month** · avg 276 input + 75 output tokens per SMS

| Option | Throughput | GPU hours / month | Cost / month (USD) | Cost / 1M SMS (USD) |
|---|---|---|---|---|
| Tiny Aya fine-tuned, self-hosted, pay-per-hour (Tesla T4 @ $0.35/h) | 1.12 SMS/s · 83 tok/s per GPU | 2,490 (≈3.4 T4s running full time) | **871.52** | **87.15** |
| gpt-5-mini API (list price; excludes reasoning tokens) | – | – | 2,198.59 | 219.86 |
| gpt-5.4-mini API (list price; excludes reasoning tokens) | – | – | 5,464.01 | 546.40 |

Throughput is measured, unoptimised Hugging Face `generate` with batch 32 on a free T4; a serving engine (vLLM/TGI) would raise it and cut the self-hosted cost further. One-time fine-tuning cost: about $0.56 (96 min on a T4).

Method: realistic volume (10,000,000 SMS/month) ÷ measured throughput (83 generated tokens/s, batched greedy decoding on a T4) → GPU hours × hourly GPU price. API cost uses list prices and our measured average prompt/answer lengths (counted with Tiny Aya's tokenizer, an approximation). GPT-5-family models may also bill hidden reasoning tokens, so API figures are a **lower bound**. We also publish an 8-bit GGUF export for **CPU** inference with llama.cpp (we have not benchmarked its CPU speed or accuracy), which makes on-device or on-premise extraction possible: a privacy win for financial data.

**Licence note:** Tiny Aya is released under CC-BY-NC. A commercial deployment needs a licence from Cohere; the cost comparison is about compute, not licensing.

**Cost per 1,000 SMS:** self-hosted fine-tuned model **$0.087** (measured throughput, T4 at $0.35/h) vs gpt-5-mini $0.220 and gpt-5.4-mini $0.546 (list prices, excluding reasoning tokens).

**Compute used:** Kaggle free tier (NVIDIA T4, 16 GB; one GPU used), plus a few failed attempts on free Colab. The final run took 4.2 GPU-hours end to end (baselines, two fine-tunes, all evaluations); all runs including debugging and failed attempts took roughly 15 T4-hours. Cost to us: $0 (free tier); at $0.35/h that is about $5.

## 9. Limitations

- **Synthetic training data.** Real SMS are more diverse than our 111 templates. The hand-written challenge set and the unseen-format split are our guard against overfitting, but they are small (44 and 300 examples). Results on real bank traffic will be lower.
- **One author wrote both the generator and the challenge set**, so some stylistic bias is unavoidable. Native-speaker review of more Hindi/Hinglish messages would strengthen it.
- **Romanised Hindi has no standard spelling.** We cover common spellings only.
- **Schema decisions are opinions:** failed transactions count as transactions with `status: failed`; future debits do not. Other apps may want different rules.
- **Single seed** (free GPU budget). Confidence intervals come from bootstrap resampling of examples, not from repeated training runs.
- **Run selection.** We trained two runs that differ only in the number of epochs (1 vs 3). We moved to 3 epochs because the 1-epoch model reproduced only 69% of its own training answers (underfitting, measured on training data). We chose the 3-epoch run with a rule fixed before seeing its test scores: it had to match or beat the 1-epoch run on dev and on unseen-format exact match. Because a test split took part in that choice, its number may be slightly optimistic. The hand-written challenge set was never used for any decision, and it improved by a similar margin (59.1% → 70.5%). Both runs are published.
- **Dates are assumed day-first** (Indian convention).
- **No general-capability check.** We did not measure whether the merged model got worse at other tasks.
- **No closed-model accuracy baseline.** The API comparison covers cost, not accuracy.
- **The GGUF export is not benchmarked** for CPU speed or accuracy.

## 10. With more credits

**Planned approach:**
1. **A real-world test set:** 300–500 real SMS donated with consent, anonymised, and labelled by two annotators (Hindi and Hinglish native speakers), to measure the gap between synthetic and real messages.
2. **More varied training data:** more templates per family, LLM-paraphrased variants checked by humans, and more wordings for credits ("salary aa gayi", "refund processed to card"), the main remaining error.
3. **Loss only on the answer** (not on the 400-token preamble), 2–3 training seeds, and the ablation and GGUF export for **Tiny Aya Fire**, which did better than Global in our comparison (section 8.4).
4. **Fast serving** with vLLM, and a CPU benchmark of the GGUF.
5. **A closed-API accuracy baseline** (e.g. gpt-5-mini) on the same test sets, and a general-capability retention check.

**Expected before → after (forecasts, not results):**
- Hand-written exact match 70.5% → about 80%, from more template variety and answer-only loss.
- Throughput 1.1 → 10+ SMS/s per T4 with vLLM, cutting self-hosted cost from about $87 to under $15 per million SMS.
- Real-SMS exact match: expected to be below the synthetic test sets at first; we would report the gap rather than guess it.

## 11. Artifacts

- **HF models collection:** https://huggingface.co/collections/Meowniece/tiny-aya-sms-extraction-models-6af9a1a0ced10d8fb22b4ed6 (LoRA adapter, GGUF, ablation adapter, Tiny Aya Fire adapter)
- **HF datasets collection:** https://huggingface.co/collections/Meowniece/tiny-aya-sms-extraction-datasets-6afa2f2876dd2d18a69a3e77 (all splits, curated training set, every prediction, metrics, training logs, AuditKIT run records, CuratorKIT provenance, figures)
- **GitHub:** https://github.com/Anshul-Amin/tiny-aya-sms-extract
- **Demo (results explorer):** https://huggingface.co/spaces/Meowniece/tiny-aya-sms-extract-v2-demo
- The `-v2` repos are this submission (3 epochs). The repos without `-v2` hold the earlier 1-epoch run, kept for transparency.

## 12. Documentation and reproduction

- **Environment:** a single NVIDIA T4 (16 GB) is enough. Kaggle image with Python 3.13; `aligntune` 0.1.35, `curatorkit`, `auditkit`, `transformers`, `trl`, `peft`, `accelerate`, `datasets`, `bitsandbytes` (installed by the notebook). A Hugging Face token with access to Tiny Aya.
- **Run:** import `notebooks/sms_extraction_pipeline.ipynb` into Kaggle, choose a GPU accelerator with internet on, add an `HF_TOKEN` secret, *Run all*. `QUICK_MODE=True` runs everything on tiny subsets (~40 min); the full run takes about 4 hours. All seeds are fixed.
- **Repo layout:**

| Path | What |
|---|---|
| `notebooks/sms_extraction_pipeline.ipynb` | The whole pipeline, self-contained |
| `src/smsgen.py` | Data generator (111 templates, 14 families × 3 languages, held-out split) |
| `src/scoring.py` | JSON parsing, schema checks, field metrics, hallucination/omission, McNemar, bootstrap |
| `src/regex_baseline.py` | Rule-based baseline |
| `src/train_worker.py` | AlignTune training in a separate process |
| `data/challenge_set.jsonl` | Hand-written challenge set |
| `space_static/`, `space/` | Static results explorer (the public Space) and a live Gradio app |
| `tests/test_core.py`, `tests/dry_run.py` | Data/scoring tests; full-pipeline CPU dry run with a fake model |
| `docs/` | This write-up and the abstract |

- **Declaration of work done outside the hackathon:** none. All code, data and models were created during the hackathon; we only used public libraries and the hackathon kits.

## 13. Team, acknowledgements and references

- **Team:** Anshul J Amin (solo).
- **Acknowledgements:** Lexsi Labs for CuratorKIT, AlignTune and AuditKIT; Cohere Labs for Tiny Aya; Kaggle for free GPUs.
- **References:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021; Hugging Face TRL and PEFT; llama.cpp; the two related projects in section 2.

*Built with Lexsi Labs' CuratorKIT, AlignTune and AuditKIT on Cohere Labs' Tiny Aya Global.*
