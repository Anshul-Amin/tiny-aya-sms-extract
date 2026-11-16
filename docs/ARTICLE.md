# Teaching a 3B model to read Indian bank SMS, in English, Hindi and Hinglish, without making things up

*Lexsi AI Safety & Alignment Hackathon · Track 1 (Fine-tuning & Data Curation) · Problem 5 (Structured output & extraction), also Problem 2 (replacing a closed-model API) and Problem 4 (multilingual & code-mixed)*

**Model:** `CohereLabs/tiny-aya-global` + LoRA · **Toolkits:** CuratorKIT → AlignTune → AuditKIT
**Links:** [LoRA adapter](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-lora) · [GGUF (CPU)](https://huggingface.co/Meowniece/tiny-aya-sms-extract-v2-gguf) · [Dataset, predictions & logs](https://huggingface.co/datasets/Meowniece/tiny-aya-sms-extract-v2-data) · [Results explorer (Space)](https://huggingface.co/spaces/Meowniece/tiny-aya-sms-extract-v2-demo) · [GitHub](https://github.com/Anshul-Amin/tiny-aya-sms-extract)

---

## TL;DR

Expense trackers, lending apps and personal-finance tools in India read billions of transaction SMS a year. Many send each message to a closed LLM API. We fine-tuned **Tiny Aya Global (3.35B)** to turn one SMS into one schema-valid JSON object (11 fields), in **English, Hindi (Devanagari) and Hinglish**. It also recognises when a message mentions money but is **not** a transaction: OTPs, offers, bill reminders, future debits and money requests.

- On message formats it **never saw in training**, the fine-tuned model reaches **85.0%** exact match (all 11 fields right), vs **21.0%** for few-shot prompting of the same model.
- On a **hand-written challenge set** of realistic, messy SMS: **70.5%** exact match vs 25.0% (few-shot), 9.1% (zero-shot) and 25.0% (a hand-written regex parser).
- **Hallucinated fields** on the challenge set: **10.1%** vs 25.3% for few-shot.
- **Ablation:** the same fine-tune *without* our robustness data (hard negatives, noise, missing fields) gets 47.7% exact match and 13.3% non-transaction F1 on the challenge set (vs 81.5%).
- **Cost at 10,000,000 SMS/month:** **$871.52** self-hosted on a T4 (2,490 GPU-hours; ≈3.4 T4s running full time), vs gpt-5-mini $2,198.59 and gpt-5.4-mini $5,464.01 per month at closed-API list prices. Fine-tuning itself cost about $0.56.

## 1. Why this problem

A bank SMS looks simple until you meet real ones. *"Rs.2,999 spent at CROMA. Avl Lmt Rs.47,001"* has two amounts, and only one is the transaction; the other is a credit limit, not a balance. *"482913 is OTP for txn of INR 12,500 at AMAZON"* has an amount, a merchant and a card, but no money has moved. *"आपके खाते XX७७१२ में ₹२५,०००.०० जमा हुए"* uses Devanagari digits. Hinglish users type *"rs.500debit hue atm se"*.

For downstream finance apps, **a wrong value is worse than a missing one**: an invented balance or a phantom transaction corrupts a budget silently. So we optimise and report not just accuracy, but **hallucination**: how often the model fills a field that the SMS does not contain.

## 2. Why this task, this model and this data

**Task.** SMS-to-JSON extraction is a high-volume, repetitive, well-defined job. That is exactly the kind of workload that is usually sent to a closed API and should not need one (Problem 2). It is checkable field by field, so improvement can be measured without an LLM judge (Problem 5). And in India it is inherently multilingual and code-mixed (Problem 4).

**Model: Tiny Aya Global (3.35B).**
- **Language balance.** It is the Tiny Aya variant tuned for balanced performance across languages and regions. Our inputs mix English, Hindi and romanised Hindi in the same message, so a balanced variant is the safer default. *Tiny Aya Fire* (South-Asia specialised) is a natural follow-up comparison.
- **Size fits the budget.** At 3.35B parameters, LoRA fine-tuning and batched inference both fit on one free 16 GB T4, and an 8-bit GGUF export can run on CPU. Aya Expanse 8B/32B would be more capable, but they cannot be fine-tuned on free hardware or served cheaply on-premise, and cheap on-premise serving is the point of the ROI argument.
- **Small output space.** The output is short and structured (~80 tokens), so model size matters less than format discipline. Fine-tuning targets format discipline directly.

**Data: synthetic, with exact labels.** See section 3 for why we generated the data and exactly how.

## 3. The dataset: why synthetic, and how it was built

### 3.1 Why we generated the data instead of collecting it

- **No usable public dataset exists.** We found no public, field-labelled dataset of Indian bank/UPI SMS, and none in Hindi or Hinglish. The closest public projects are a regex library ([transaction-sms-parser](https://github.com/saurabhgupta050890/transaction-sms-parser), MIT) that ships a single example message, and a GPT-2 parser ([banking-sms-json-parser-v8](https://huggingface.co/rawsun00001/banking-sms-json-parser-v8)) trained on 60k synthetic English messages whose data is not released.
- **Real SMS are personal financial data.** They contain account digits, balances, names and UPI IDs. Collecting them from people's phones and publishing them would need informed consent under India's Digital Personal Data Protection Act, 2023, and could not be released as an open dataset. The hackathon also asks us to host the data publicly.
- **Exact labels make the safety metric possible.** Hand-labelling real SMS is slow and noisy. In particular, nobody can say with certainty whether an annotator "invented" a field. With generated data, every label is known, so we can measure hallucination exactly.
- **Control.** We can create the hard cases on purpose, such as OTPs that mention amounts or credit limits next to the real amount, and hold out entire message formats to test generalisation.

The risk of synthetic data is that a model learns the templates instead of the task. Sections 3.3 and 5 show how we test for that: a split made only of unseen formats, and a hand-written set from outside the generator.

### 3.2 How each message is built

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

### 3.3 Splits that test generalisation, not memorisation

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

### 3.4 Curation and validation

- **CuratorKIT.** The training split is exported in Alpaca format and run through CuratorKIT for cleaning, exact de-duplication and provenance (manifest, checksums, `lexsi_provenance.json`, dataset card). Curation report: 1,600 rows in → CuratorKIT exact de-dup exported 1,108 (its exact matching ignores the SMS text, so different messages with identical labels collapse) → our JSON gate restored the 492 valid rows → 1,600 final; MinHash would have kept only 25.
- **Our JSON quality gate.** After curation, every row's output must still parse to JSON identical to its gold label, or it is dropped.
- **Why no near-duplicate removal.** We ran MinHash near-duplicate detection and report its count, but did not apply it: templated SMS are near-duplicates *by design* yet carry different labels, so removing them would delete exactly the minimal pairs the model needs.
- **Automated checks** (`tests/test_core.py`, run before every training run):
  - every gold value is grounded in the SMS text;
  - all 2,364 gold labels are schema-valid and score 100% against themselves (checked during the audit);
  - no SMS appears in more than one split;
  - the challenge set is valid.

## 4. Systems compared

| System | What it is |
|---|---|
| `regex` | A careful hand-written rule parser (English + Hindi/Hinglish keywords, Indian grouping, day-first dates). A strong non-ML baseline, not a straw man. |
| `zero_shot` | Tiny Aya Global with the schema instructions only |
| `few_shot` | Same + 3 fixed exemplars from train (EN, HI, Hinglish negative) |
| `ft_no_robust_data` | LoRA fine-tune on the same number of **clean** examples: no hard negatives, no noise, no missing fields (ablation) |
| `fine_tuned` | **Ours:** LoRA fine-tune on the curated robustness data |

Fine-tuning: **AlignTune** `create_sft_trainer` (TRL backend; path used: `aligntune`), LoRA r=16, α=32, lr=2e-4, 3 epochs, effective batch 16, max length 832 tokens, 96 min on a single free T4. Greedy decoding. The prompt format (`chat`) was selected on dev only.

**Evaluation.** Every system on every split was run through **AuditKIT** (run records and fingerprints are in the dataset repo). Training logs and loss curves are in the model repo under `logs/`.

### Metrics, and why each one

| Metric | Definition | Why it matters for this use case |
|---|---|---|
| **Exact match** | all 11 fields correct | An app consumes the whole record; one wrong field (e.g. debit vs credit) breaks the budget. This is the headline metric. |
| **Field accuracy** | per-field correctness on transaction SMS | Shows *where* errors are: amount vs counterparty vs date. |
| **Hallucination rate ↓** | non-null prediction where gold is null | In finance a made-up balance or reference is worse than a blank; this is the reliability metric. |
| **Omission rate ↓** | null prediction where gold has a value | The other side of the trade-off: a model that always says null would have zero hallucination. |
| **Non-transaction F1** | catching OTPs, offers, reminders, future debits | A phantom transaction is the most damaging error for downstream apps. |
| **Schema validity** | parseable JSON, right keys and types | The integration contract: invalid output needs retries or a human. |

Matching is lenient on formatting (case, whitespace, `1,23,456` vs `123456`, Devanagari digits) and strict on meaning. Significance uses a paired exact McNemar test; intervals use bootstrap resampling.

## 5. Results

### Seen formats (test_id)
| System | Exact match | Field acc. | Halluc. ↓ | Omission ↓ | Non-txn F1 | Schema valid |
|---|---|---|---|---|---|---|
| regex | 31.0 | 75.6 | 12.9 | 27.6 | 67.0 | 100.0 |
| zero_shot | 5.7 | 79.5 | 65.4 | 3.1 | 60.2 | 86.0 |
| few_shot | 26.0 | 84.0 | 25.3 | 14.5 | 67.9 | 85.7 |
| ft_no_robust_data | 59.7 | 97.7 | 49.8 | 0.2 | 6.9 | 96.3 |
| **fine_tuned** | 92.0 | 98.7 | 0.2 | 0.3 | 100.0 | 100.0 |

### Unseen formats (test_ood)
| System | Exact match | Field acc. | Halluc. ↓ | Omission ↓ | Non-txn F1 | Schema valid |
|---|---|---|---|---|---|---|
| regex | 46.7 | 91.3 | 4.2 | 9.9 | 92.7 | 100.0 |
| zero_shot | 10.0 | 82.3 | 51.7 | 1.9 | 81.9 | 82.0 |
| few_shot | 21.0 | 87.3 | 21.3 | 11.7 | 83.2 | 88.3 |
| ft_no_robust_data | 50.7 | 96.8 | 42.9 | 0.7 | 46.8 | 94.0 |
| **fine_tuned** | 85.0 | 97.8 | 2.0 | 0.2 | 98.2 | 99.7 |

### Hand-written challenge set
| System | Exact match | Field acc. | Halluc. ↓ | Omission ↓ | Non-txn F1 | Schema valid |
|---|---|---|---|---|---|---|
| regex | 25.0 | 87.0 | 24.2 | 10.7 | 57.1 | 100.0 |
| zero_shot | 9.1 | 78.7 | 63.1 | 2.5 | 52.6 | 70.5 |
| few_shot | 25.0 | 84.7 | 25.3 | 15.3 | 80.0 | 86.4 |
| ft_no_robust_data | 47.7 | 96.7 | 46.5 | 0.4 | 13.3 | 97.7 |
| **fine_tuned** | 70.5 | 92.0 | 10.1 | 6.6 | 81.5 | 100.0 |

### Is the improvement real? (paired exact McNemar test on exact match)
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

### By language: exact match on unseen formats
| System | en EM | hi EM | hinglish EM |
|---|---|---|---|
| regex | 63.3 | 36.7 | 40.9 |
| zero_shot | 11.2 | 11.0 | 7.5 |
| few_shot | 21.4 | 10.1 | 33.3 |
| ft_no_robust_data | 51.0 | 55.0 | 45.2 |
| **fine_tuned** | **83.7** | **90.8** | **79.6** |

### By language: exact match on the challenge set
| System | en EM (n=19) | hi EM (n=12) | hinglish EM (n=13) |
|---|---|---|---|
| regex | 26.3 | 25.0 | 23.1 |
| zero_shot | 15.8 | 0.0 | 7.7 |
| few_shot | 31.6 | 16.7 | 23.1 |
| ft_no_robust_data | 36.8 | 50.0 | 61.5 |
| **fine_tuned** | **57.9** | **75.0** | **84.6** |

### Edge cases: exact match on the challenge set
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

## 6. What we learned

- **The model learned the task, not the templates.** On message formats it never saw, exact match is 85.0% versus 92.0% on seen formats: a 7-point gap. Hindi is the strongest language on unseen formats (90.8%).
- **Prompting alone is not reliable enough to ship.** Zero-shot output was valid JSON only 82.0% of the time on unseen formats and hallucinated 52% of null fields; few-shot still hallucinated 21%. The fine-tuned model is valid 99.7% of the time and hallucinates 2.0%.
- **The robustness data is what makes it safe.** Trained on the same amount of clean data, the ablation scores well on field accuracy but cannot tell an OTP, offer or future debit from a real transaction (non-transaction F1 46.8% vs 98.2% on unseen formats) and hallucinates 43% of null fields. The improvement is significant on every split (McNemar p ≤ 0.021).
- **Training long enough mattered as much as the data.** With one epoch the model reproduced only 69% of its own training answers (dev exact match 70%); three epochs reached 95.8% on dev, 92.0 / 85.0 / 70.5% on the test sets. Both runs are published.
- **Check what the trainer actually sees.** Tiny Aya's chat template adds a ~400-token default preamble, so our first run (max length 512) silently cut every JSON answer off: 99.8% token accuracy, ~2% exact match. A sanity check (can the model reproduce 16 training answers?) and a log of the last tokens of a training example now guard against this.
- **Still hard:** promotional text appended to a real transaction, Devanagari-digit account numbers and references, and English messages in the hand-written set (57.9% vs 75–85% for Hindi/Hinglish).
