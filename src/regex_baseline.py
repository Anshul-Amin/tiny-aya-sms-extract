"""A reasonable hand-written rule-based parser: the 'no ML' baseline.

It is deliberately decent (not a straw man): it handles the common English formats, the
usual Hindi/Hinglish keywords, Indian digit grouping and day-first dates.
"""
from __future__ import annotations

import json
import re

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_DEV = str.maketrans("०१२३४५६७८९", "0123456789")

NEG_HINTS = ["otp", "will be debited", "due on", "cashback", "pre-approved", "offer",
             "debit honge", "डेबिट किए जाएँगे", "कटेगी", "kategi", "देय है", "due hai",
             "upcoming", "आगामी", "agli emi", "balance in a/c", "avl bal in your", "शेष राशि", "उपलब्ध शेष"]
DEBIT_HINTS = ["debited", "debit", "spent", "withdrawn", "sent", "paid", "डेबिट", "खर्च", "निकाले",
               "निकासी", "भेजे गए", "कटे", "kharch", "nikale", "bheje gaye", "kat gaye", "withdraw",
               "used for", "चुकाए", "pay hue", "declined", "failed", "विफल", "fail", "nahi ho"]
CREDIT_HINTS = ["credited", "received", "refund", "reversed", "जमा", "प्राप्त", "रिफ़ंड", "वापस",
                "credit hue", "mile", "aaya", "wapas", "bheje{@}", "ने आपको", "ne aapko", "salary"]
AMT_RE = re.compile(
    r"(?:(?:rs\.?|inr|₹|रु\.?|रुपये|usd|eur|gbp|aed|\$|€|£)\s*([\d,]+(?:\.\d+)?))|(?:([\d,]+(?:\.\d+)?)\s*rupaye)",
    re.I)


def _num(s):
    return float(s.replace(",", ""))


def _date(t: str):
    t = t.lower()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"\b(\d{1,2})[-/. ]?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/. ]?(\d{2,4})\b", t)
    if m:
        y = int(m.group(3)); y = y + 2000 if y < 100 else y
        return f"{y:04d}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", t)
    if m:
        y = int(m.group(3)); y = y + 2000 if y < 100 else y
        return f"{y:04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return None


def parse(sms: str) -> dict:
    t = sms.translate(_DEV)
    low = t.lower()
    out = {k: None for k in ["is_transaction", "type", "amount", "currency", "account_last4",
                             "instrument", "counterparty", "date", "reference", "balance", "status"]}
    if any(h in low for h in NEG_HINTS) and not any(h in low for h in ["debited from", "credited"]):
        out["is_transaction"] = False
        return out
    out["is_transaction"] = True
    is_credit = any(h in low for h in CREDIT_HINTS)
    out["type"] = "credit" if is_credit else "debit"
    amts = list(AMT_RE.finditer(t))
    if amts:
        m = amts[0]
        out["amount"] = _num(m.group(1) or m.group(2))
        head = m.group(0).lower()
        out["currency"] = ("USD" if ("usd" in head or "$" in head) else "EUR" if ("eur" in head or "€" in head)
                           else "GBP" if ("gbp" in head or "£" in head) else "AED" if "aed" in head else "INR")
    m = re.search(r"(?:bal(?:ance)?|शेष(?: राशि)?)\s*:?\s*(?:is\s*)?(?:rs\.?|inr|₹|रु\.?|रुपये)?\s*([\d,]+(?:\.\d+)?)", t, re.I)
    if m:
        out["balance"] = _num(m.group(1))
    m = re.search(r"(?:x+|\*+|ending\s|-)(\d{4})\b", t, re.I)
    if m:
        out["account_last4"] = m.group(1)
    for inst in ["NEFT", "IMPS", "RTGS", "NACH", "UPI", "ATM"]:
        if inst.lower() in low:
            out["instrument"] = inst
            break
    if "autopay" in low or "ऑटोपे" in low:
        out["instrument"] = "AUTOPAY"
    if "card" in low or "कार्ड" in low:
        out["instrument"] = "CARD"
    m = re.search(r"(?:ref(?:erence)?\.?\s*(?:no\.?)?|utr|संदर्भ(?: संख्या)?|रेफ़)\s*:?\s*([A-Z0-9]{8,})", t, re.I)
    if m:
        out["reference"] = m.group(1)
    m = re.search(r"(?:to|at|from|by|towards)\s+(?:vpa\s+)?([A-Za-z0-9.@ ]+?)(?=\s+(?:on|via|for|ke|ko|se|par)\b|[.,;(]|$)", t)
    if m:
        out["counterparty"] = m.group(1).strip()
    out["date"] = _date(t)
    out["status"] = ("failed" if any(h in low for h in ["fail", "declined", "विफल", "अस्वीकृत", "could not", "nahi ho", "नहीं हो सका", "decline"])
                     else "reversed" if any(h in low for h in ["reversed", "वापस कर दिए", "wapas aa gaye"])
                     else "success")
    return out


def predict(sms: str) -> str:
    return json.dumps(parse(sms), ensure_ascii=False)
