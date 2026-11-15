---
title: Tiny Aya SMS to JSON
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Indian bank/UPI SMS to JSON in English, Hindi, Hinglish
---

# Tiny Aya · Indian bank & UPI SMS → JSON

Fine-tuned Tiny Aya Global extracts 11 fields (type, amount, currency, account, instrument, counterparty, date, reference, balance, status) from bank and UPI SMS in English, Hindi and Hinglish. It flags non-transactions such as OTPs, offers, reminders and future debits.

- Model (LoRA): https://huggingface.co/__MODEL_REPO__
- Model (GGUF, used here): https://huggingface.co/__GGUF_REPO__
- Data, predictions & metrics: https://huggingface.co/datasets/__DATA_REPO__

Built for the Lexsi AI Safety & Alignment Hackathon (Track 1, Problem 5) with CuratorKIT, AlignTune and AuditKIT.
