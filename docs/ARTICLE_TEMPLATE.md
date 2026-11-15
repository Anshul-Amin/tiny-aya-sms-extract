# Teaching a 3B model to read Indian bank SMS, in English, Hindi and Hinglish, without making things up

*Lexsi AI Safety & Alignment Hackathon · Track 1 (Fine-tuning & Data Curation) · Problem 5 (Structured output & extraction), also Problem 2 (replacing a closed-model API) and Problem 4 (multilingual & code-mixed)*

**Model:** `{{BASE_MODEL}}` + LoRA · **Toolkits:** CuratorKIT → AlignTune → AuditKIT
**Links:** [LoRA adapter](https://huggingface.co/{{MODEL_REPO}}){{GGUF_LINK}} · [Dataset, predictions & logs](https://huggingface.co/datasets/{{DATA_REPO}}) · [Results explorer (Space)](https://huggingface.co/spaces/{{SPACE_REPO}}) · [GitHub](https://github.com/Anshul-Amin/tiny-aya-sms-extract)

---

## TL;DR

Expense trackers, lending apps and personal-finance tools in India read billions of transaction SMS a year. Many send each message to a closed LLM API. We fine-tuned **Tiny Aya Global (3.35B)** to turn one SMS into one schema-valid JSON object (11 fields), in **English, Hindi (Devanagari) and Hinglish**. It also recognises when a message mentions money but is **not** a transaction: OTPs, offers, bill reminders, future debits and money requests.

- On message formats it **never saw in training**, the fine-tuned model reaches **{{FT_EM_OOD}}%** exact match (all 11 fields right), vs **{{FS_EM_OOD}}%** for few-shot prompting of the same model.
- On a **hand-written challenge set** of realistic, messy SMS: **{{FT_EM_CH}}%** exact match vs {{FS_EM_CH}}% (few-shot), {{ZS_EM_CH}}% (zero-shot) and {{RX_EM_CH}}% (a hand-written regex parser).
- **Hallucinated fields** on the challenge set: **{{FT_HALL_CH}}%** vs {{FS_HALL_CH}}% for few-shot.
- **Ablation:** the same fine-tune *without* our robustness data (hard negatives, noise, missing fields) gets {{ABL_EM_CH}}% exact match and {{ABL_NT_CH}}% non-transaction F1 on the challenge set (vs {{FT_NT_CH}}%).
- **Cost at {{MONTHLY_SMS}} SMS/month:** **${{SELF_MONTH}}** self-hosted on a T4 ({{GPU_HOURS}} GPU-hours; ${{ALWAYS_ON_MONTH}} for an always-on GPU), vs {{API_MONTH}} per month at closed-API list prices. Fine-tuning itself cost about ${{TRAIN_COST}}.

## 1. Why this problem

A bank SMS looks simple until you meet real ones. *"Rs.2,999 spent at CROMA. Avl Lmt Rs.47,001"* has two amounts, and only one is the transaction; the other is a credit limit, not a balance. *"482913 is OTP for txn of INR 12,500 at AMAZON"* has an amount, a merchant and a card, but no money has moved. *"आपके खाते XX७७१२ में ₹२५,०००.०० जमा हुए"* uses Devanagari digits. Hinglish users type *"rs.500debit hue atm se"*.

For downstream finance apps, **a wrong value is worse than a missing one**: an invented balance or a phantom transaction corrupts a budget silently. So we optimise and report not just accuracy, but **hallucination**: how often the model fills a field that the SMS does not contain.

## 2. Why this task, this model and this data

**Task.** SMS-to-JSON extraction is a high-volume, repetitive, well-defined job. That is exactly the kind of workload that is usually sent to a closed API and should not need one (Problem 2). It is checkable field by field, so improvement can be measured without an LLM judge (Problem 5). And in India it is inherently multilingual and code-mixed (Problem 4).

**Model: Tiny Aya Global (3.35B).**
- **Language balance.** It is the Tiny Aya variant tuned for balanced performance across languages and regions. Our inputs mix English, Hindi and romanised Hindi in the same message, so a balanced variant is the safer default. *Tiny Aya Fire* (South-Asia specialised) is a natural follow-up comparison.
- **Size fits the budget.** At 3.35B parameters, LoRA fine-tuning and batched inference both fit on one free 16 GB T4, and an 8-bit GGUF export can run on CPU. Aya Expanse 8B/32B would be more capable, but they cannot be fine-tuned on free hardware or served cheaply on-premise, and cheap on-premise serving is the point of the ROI argument.
- **Small output space.** The output is short and structured (~80 tokens), so model size matters less than format discipline. Fine-tuning targets format discipline directly.

**Data: synthetic, with exact labels.** Real bank SMS contain account numbers, balances and names, so they cannot be collected or published responsibly. Synthetic generation gives (a) **exact gold labels**, (b) **full control over edge cases**, and (c) a **publishable dataset**. The risk of synthetic data is overfitting to templates. We counter it with a template-level held-out split and a hand-written challenge set.

## 3. Data: exact labels by construction

We wrote a generator (`smsgen.py`) with **14 message families × 3 languages**: UPI debit/credit, card spend (incl. foreign currency), ATM, NEFT/IMPS/RTGS in and out, refunds & reversals, failed transactions, EMI/AutoPay/NACH, and **5 hard-negative families** (OTP, promo, bill due, scheduled debit, balance info).

Key design choices:

1. **Segment templates → exact gold.** Each template is split into segments. Optional segments are randomly dropped and messages are sometimes truncated. A field is labelled **only if its text survived**, so the gold never asks the model to know something that is not in the message. This is what makes the hallucination metric meaningful.
2. **Template-level held-out split.** For every (family, language), one template is never used for training. `test_ood` contains only those formats, so it measures **generalisation to unseen message formats**, not memorisation of templates.
3. **Edge cases as first-class data:** Indian digit grouping (₹1,23,456.50), 8 date formats (day-first), masked accounts (`XX1234`, `*1234`, `ending 1234`), credit-limit distractors, promo tails with extra amounts, ALL CAPS / lowercase, missing spaces (`Rs.450debited`), Devanagari numerals, sender headers and footers.
4. **Responsible synthetic data:** all bank names are fictional and messages contain no links or phone numbers, so the dataset can't be confused with, or reused as, real bank messages. No real customer data was used.
5. **Hand-written challenge set ({{N_CHALLENGE}} SMS):** written and labelled by hand in styles the generator never produces (reordered fields, pipe-separated alerts, "Cr/Dr" abbreviations, money requests, cashback credits, mixed scripts).

| Split | Size | Purpose |
|---|---|---|
| train (after CuratorKIT) | {{N_TRAIN}} | fine-tuning |
| dev | {{N_DEV}} | prompt-format choice only |
| test_id | {{N_TEST_ID}} | seen formats, new values |
| test_ood | {{N_TEST_OOD}} | **unseen formats** |
| challenge | {{N_CHALLENGE}} | **hand-written, messy** |

**CuratorKIT.** The training split is exported in Alpaca format and run through CuratorKIT (cleaning, exact de-duplication, provenance: manifest, checksums, `lexsi_provenance.json`). We then add our own **quality gate**: every curated row's output must still parse to JSON identical to its gold label, otherwise it is dropped. We also ran MinHash near-duplicate detection and report its count, but did not apply it: templated SMS are near-duplicates *by design* yet carry different labels, so near-dup removal would delete exactly the minimal pairs the model needs. Curation report: `{{CURATION}}`

## 4. Systems compared

| System | What it is |
|---|---|
| `regex` | A careful hand-written rule parser (English + Hindi/Hinglish keywords, Indian grouping, day-first dates). A strong non-ML baseline, not a straw man. |
| `zero_shot` | Tiny Aya Global with the schema instructions only |
| `few_shot` | Same + 3 fixed exemplars from train (EN, HI, Hinglish negative) |
| `ft_no_robust_data` | LoRA fine-tune on the same number of **clean** examples: no hard negatives, no noise, no missing fields (ablation) |
| `fine_tuned` | **Ours:** LoRA fine-tune on the curated robustness data |

Fine-tuning: **AlignTune** `create_sft_trainer` (TRL backend; path used: `{{TRAIN_PATH}}`), LoRA r={{LORA_R}}, α={{LORA_ALPHA}}, lr={{LR}}, {{EPOCHS}} epoch, {{TRAIN_MIN}} min on a single free T4. Greedy decoding. The prompt format (`{{FT_FORMAT}}`) was selected on dev only.

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
{{TABLE_TEST_ID}}

### Unseen formats (test_ood)
{{TABLE_TEST_OOD}}

### Hand-written challenge set
{{TABLE_CHALLENGE}}

### Is the improvement real? (paired exact McNemar test on exact match)
{{SIG_TABLE}}

### By language: exact match on unseen formats
{{LANG_TABLE_OOD}}

### By language: exact match on the challenge set
{{LANG_TABLE_CHALLENGE}}

### Edge cases: exact match on the challenge set
{{TAG_TABLE_CHALLENGE}}

![Exact match](https://huggingface.co/datasets/{{DATA_REPO}}/resolve/main/figures/exact_match.png)
![Hallucination](https://huggingface.co/datasets/{{DATA_REPO}}/resolve/main/figures/hallucination.png)
![Non-transaction F1](https://huggingface.co/datasets/{{DATA_REPO}}/resolve/main/figures/nontxn_f1.png)

## 6. What we learned

> *[EDIT after the run: 3–5 short bullet points in your own words. Prompts to answer from the tables above:]*
> - *How large is the gap between seen (test_id) and unseen formats (test_ood)? A small gap means the model learned the task, not the templates.*
> - *Did prompting alone produce valid JSON reliably? (zero-shot schema validity on test_ood: {{ZS_SCHEMA_OOD}}% vs fine-tuned {{FT_SCHEMA_OOD}}%)*
> - *What did the robustness data buy? Compare `ft_no_robust_data` vs `fine_tuned` on non-transaction F1 and hallucination.*
> - *Which language is hardest, and which edge case still fails most?*

Most frequent remaining errors (fine-tuned, unseen + challenge): {{WRONG_FIELDS}}. Every error, with gold and prediction, is in `results/error_analysis_fine_tuned.csv`.

## 7. What this means in practice

- **Drop-in for an API call:** same input (one SMS), same output (one JSON object); the schema-validity rate shows how often a retry would be needed.
- **Safer by default:** a lower hallucination rate and higher non-transaction F1 mean fewer phantom transactions and invented balances in users' budgets.
- **Works for how Indians actually text:** Hindi script, Hinglish, Devanagari digits and Indian number grouping are supported natively, with no transliteration step.
- **Private:** runs on-premise or on a CPU, so no financial SMS leaves the company's infrastructure.

## 8. ROI: replacing a closed-model API

{{ROI_TABLE}}

Method: realistic volume ({{MONTHLY_SMS}} SMS/month) ÷ measured throughput ({{TOK_S}} generated tokens/s, batched greedy decoding on a T4) → GPU hours × hourly GPU price. API cost uses list prices and our measured average prompt/answer lengths (counted with Tiny Aya's tokenizer, an approximation). GPT-5-family models may also bill hidden reasoning tokens, so API figures are a **lower bound**. The fine-tuned model can also be exported to an 8-bit GGUF for **CPU** inference (llama.cpp), so on-device or on-prem extraction is possible and no SMS ever leaves the user's infrastructure. That is a privacy win for financial data.

**Licence note:** Tiny Aya is released under CC-BY-NC. A commercial deployment needs a licence from Cohere; the cost comparison is about compute, not licensing.

## 9. Limitations

- **Synthetic training data.** Real SMS are more diverse than our 111 templates. The hand-written challenge set and the unseen-format split are our guard against overfitting, but they are small ({{N_CHALLENGE}} and {{N_TEST_OOD}} examples). Results on real bank traffic will be lower.
- **One author wrote both the generator and the challenge set**, so some stylistic bias is unavoidable. Native-speaker review of more Hindi/Hinglish messages would strengthen it.
- **Romanised Hindi has no standard spelling.** We cover common spellings only.
- **Schema decisions are opinions:** failed transactions count as transactions with `status: failed`; future debits do not. Other apps may want different rules.
- **Single seed, single run** (free GPU budget). Confidence intervals come from bootstrap resampling of examples, not from repeated training runs.
- **Dates are assumed day-first** (Indian convention).

## 10. Reproduce

Open the notebook on Kaggle (T4, internet on), add an `HF_TOKEN` secret, *Run all*. `QUICK_MODE=True` runs the whole pipeline on tiny subsets in about 10 minutes. All seeds are fixed. Every split, prediction, metric file, CuratorKIT provenance file and AuditKIT run record is in the dataset repo.

*Built with Lexsi Labs' CuratorKIT, AlignTune and AuditKIT on Cohere Labs' Tiny Aya Global.*
