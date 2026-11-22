"""Stage 2 data tools: LLM rewrites of training SMS, rephrased test sets, an unrelated-text set,
and the checks that keep labels correct.

A rewrite changes the wording of an SMS but must keep every fact. Because the gold label is
reused unchanged, a rewrite is only accepted if the facts can be verified in its text:

* fact gate      - every non-null gold value (amount, balance, account, reference, counterparty,
                   date, currency, instrument keyword) is still present, and no new number appears
* meaning gate   - a rule-based reading (regex baseline) of the rewrite agrees with the original
                   on is_transaction / type / status, so a debit cannot silently become a credit
* language gate  - Hindi stays in Devanagari, English/Hinglish stay in Latin script
* hygiene        - length bounds, no links, not identical to the source or to any other split
"""
from __future__ import annotations

import json
import random
import re
import unicodedata

import regex_baseline as RB

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
URL_RE = re.compile(r"https?://|www\.|\.com\b|\.in/|bit\.ly", re.I)
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
PLACEHOLDER_RE = re.compile(r"bank name|your bank|बैंक नाम|bank ka naam|xxx bank", re.I)

INSTRUMENT_KEYWORDS = {
    "UPI": ["upi", "यूपीआई", "vpa", "@"],
    "CARD": ["card", "कार्ड"],
    "ATM": ["atm", "एटीएम"],
    "NEFT": ["neft", "एनईएफटी"],
    "IMPS": ["imps", "आईएमपीएस"],
    "RTGS": ["rtgs", "आरटीजीएस"],
    "NACH": ["nach", "ecs", "mandate", "मैंडेट"],
    "AUTOPAY": ["autopay", "auto-pay", "auto pay", "ऑटोपे", "standing instruction", "si "],
}
FOREIGN = {"USD": ["usd", "$"], "EUR": ["eur", "€"], "GBP": ["gbp", "£"], "AED": ["aed", "dhs"],
           "SGD": ["sgd", "s$"]}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").translate(DEV_DIGITS)
    return re.sub(r"\s+", " ", t).strip()


def numbers(t: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(norm(t)):
        s = m.group(0).replace(",", "").rstrip(".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def has_number(t: str, v: float) -> bool:
    return any(abs(n - float(v)) < 0.005 for n in numbers(t))


def _date_tokens(sms: str) -> list[str]:
    t = norm(sms)
    pats = [r"\d{4}-\d{2}-\d{2}", r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",
            r"\d{1,2}[-/. ]?[A-Za-z]{3,9}[-/., ]{0,2}\d{2,4}"]
    toks = []
    for p in pats:
        toks += [m.group(0) for m in re.finditer(p, t)]
    return toks


def fact_gate(orig_sms: str, new_sms: str, gold: dict) -> tuple[bool, str]:
    """True if every gold fact is still verifiable in new_sms and no new number was introduced."""
    o, n = norm(orig_sms), norm(new_sms)
    nl = n.casefold()
    for k in ("amount", "balance"):
        if gold.get(k) is not None and not has_number(n, gold[k]):
            return False, f"lost_{k}"
    for k in ("account_last4", "reference"):
        v = gold.get(k)
        if v is not None and norm(str(v)).casefold() not in nl:
            return False, f"lost_{k}"
    cp = gold.get("counterparty")
    if cp is not None and norm(cp).casefold() not in nl:
        return False, "lost_counterparty"
    if gold.get("date") is not None:
        same_parse = RB._date(n) == gold["date"]
        verbatim = any(tok.casefold() in nl for tok in _date_tokens(o))
        if not (same_parse or verbatim):
            return False, "lost_date"
    cur = gold.get("currency")
    if cur and cur != "INR":
        if not any(s in nl for s in FOREIGN.get(cur, [cur.lower()])):
            return False, "lost_currency"
    else:
        ol = o.casefold()
        for code, syms in FOREIGN.items():
            if any(s in nl and s not in ol for s in syms if s not in ("$",)):
                return False, f"new_currency_{code}"
    inst = gold.get("instrument")
    if inst:
        kws = [w for w in INSTRUMENT_KEYWORDS.get(inst, []) if w in o.casefold()]
        if kws and not any(w in nl for w in kws):
            return False, "lost_instrument"
    orig_nums = numbers(o)
    for v in numbers(n):
        if v >= 10 and not any(abs(v - x) < 0.005 for x in orig_nums):
            return False, "new_number"
    return True, "ok"


def meaning_gate(orig_sms: str, new_sms: str) -> tuple[bool, str]:
    """Rule-based reading must agree on is_transaction / type / status."""
    try:
        a, b = RB.parse(orig_sms), RB.parse(new_sms)
    except Exception:          # the rule-based reader cannot parse it: we cannot verify the meaning, so reject
        return False, "meaning_unverifiable"
    for k in ("is_transaction", "type", "status"):
        if a.get(k) != b.get(k):
            return False, f"meaning_changed_{k}"
    return True, "ok"


def language_gate(lang: str, new_sms: str) -> tuple[bool, str]:
    dev, lat = len(DEVANAGARI_RE.findall(new_sms)), len(LATIN_RE.findall(new_sms))
    letters = max(dev + lat, 1)
    if lang == "hi" and dev / letters < 0.3:
        return False, "lang_not_devanagari"
    if lang == "en" and dev / letters > 0.05:
        return False, "lang_has_devanagari"
    if lang == "hinglish" and dev / letters > 0.25:
        return False, "lang_mostly_devanagari"
    return True, "ok"


def hygiene_gate(orig_sms: str, new_sms: str, existing: set[str]) -> tuple[bool, str]:
    n = norm(new_sms)
    if not (15 <= len(n) <= 420):
        return False, "bad_length"
    if URL_RE.search(n):
        return False, "has_link"
    if n.casefold() == norm(orig_sms).casefold():
        return False, "unchanged"
    o = norm(orig_sms)
    if any(ch in n and ch not in o for ch in "[]<>{}") or PLACEHOLDER_RE.search(n):
        return False, "placeholder"
    if n.casefold() in existing:
        return False, "duplicate_or_leak"
    return True, "ok"


def check_rewrite(row: dict, new_sms: str, existing: set[str]) -> tuple[bool, str]:
    """Run all gates in a fixed order; return (accepted, first failing reason)."""
    for gate in (lambda: hygiene_gate(row["sms"], new_sms, existing),
                 lambda: language_gate(row["lang"], new_sms),
                 lambda: fact_gate(row["sms"], new_sms, row["gold"]),
                 lambda: meaning_gate(row["sms"], new_sms)):
        try:
            ok, why = gate()
        except Exception as e:  # never let one odd rewrite stop the run
            return False, f"check_error_{type(e).__name__}"
        if not ok:
            return False, why
    return True, "ok"


def clean_llm_text(t: str) -> str:
    """Strip quotes, labels and extra lines an LLM may wrap around the SMS."""
    t = (t or "").strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t).strip()
    t = re.sub(r"^(rewritten|rephrased|paraphrased)?\s*(sms|message)\s*:\s*", "", t, flags=re.I).strip()
    lines = [l for l in t.splitlines() if l.strip()]
    t = lines[0].strip() if lines else ""
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t.strip("“”\"").strip()


LANG_NAME = {"en": "English", "hi": "Hindi written in Devanagari script",
             "hinglish": "Hinglish (Hindi written in Latin/Roman script, mixed with English)"}

STYLES = [
    "terse, heavily abbreviated (A/c, Txn, Ref, Bal), like a busy bank's alert",
    "formal and polite, starting with 'Dear Customer' or its equivalent",
    "put the amount first, then the account, then the other details",
    "UPI-app notification style, short and friendly",
    "the bank name at the end after a dash, details in a different order",
    "slightly longer, adding a generic closing line such as 'not you? contact your bank' (no phone numbers, no links)",
]

TRAIN_REWRITE_SYSTEM = (
    "You rewrite Indian bank and UPI SMS so that they look like a different bank sent them. "
    "Rules: (1) Copy every number, amount, account digit, card digit, reference/UTR number, date and "
    "merchant/person name EXACTLY as written, character for character. (2) Keep the meaning exactly: "
    "a debit stays a debit, a credit stays a credit, a failed or reversed transaction stays failed or "
    "reversed, an OTP/offer/reminder/future debit/balance alert stays that and must not become a "
    "transaction. (3) Keep the same language and script. (4) Do not add any new numbers, links, phone "
    "numbers or details. (5) Reply with the rewritten SMS only, on one line, without quotes."
)

TEST_REPHRASE_SYSTEM = (
    "Paraphrase the SMS below. Change the wording and sentence order the way another bank might phrase "
    "it, but keep every fact. Every number, amount, account or card digit, reference number, date and "
    "name must appear exactly as in the original. Do not change what happened (debit/credit, success/"
    "failure, OTP or offer stays the same kind of message). Keep the language and script. Add nothing "
    "new. Output only the paraphrased SMS on one line."
)


def train_rewrite_messages(row: dict, rng: random.Random) -> list[dict]:
    style = rng.choice(STYLES)
    user = (f"Language: {LANG_NAME.get(row['lang'], row['lang'])}\nStyle: {style}\n\n"
            f"Original SMS:\n{row['sms']}")
    return [{"role": "system", "content": TRAIN_REWRITE_SYSTEM}, {"role": "user", "content": user}]


def test_rephrase_messages(row: dict) -> list[dict]:
    user = f"Language: {LANG_NAME.get(row['lang'], row['lang'])}\n\nSMS:\n{row['sms']}"
    return [{"role": "system", "content": TEST_REPHRASE_SYSTEM}, {"role": "user", "content": user}]


NULL_GOLD = {"is_transaction": False, "type": None, "amount": None, "currency": None,
             "account_last4": None, "instrument": None, "counterparty": None, "date": None,
             "reference": None, "balance": None, "status": None}


def sentences(text: str, lo: int = 30, hi: int = 220) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", re.sub(r"\s+", " ", text or "").strip())
    return [p for p in parts if lo <= len(p) <= hi]
