"""Parsing, schema validation and metrics for SMS -> JSON extraction."""
from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import defaultdict

FIELDS = ["is_transaction", "type", "amount", "currency", "account_last4", "instrument",
          "counterparty", "date", "reference", "balance", "status"]
VALUE_FIELDS = FIELDS[1:]
ENUMS = {
    "type": {"debit", "credit"},
    "instrument": {"UPI", "CARD", "NEFT", "IMPS", "RTGS", "ATM", "NACH", "AUTOPAY"},
    "status": {"success", "failed", "reversed"},
}
_DEV = str.maketrans("०१२३४५६७८९", "0123456789")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """Return the first JSON object in `text` (tolerates code fences / chatter)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def schema_errors(obj: dict | None) -> list[str]:
    if obj is None:
        return ["not_json"]
    errs = []
    keys = set(obj)
    if keys != set(FIELDS):
        if set(FIELDS) - keys:
            errs.append("missing_keys")
        if keys - set(FIELDS):
            errs.append("extra_keys")
    if not isinstance(obj.get("is_transaction"), bool):
        errs.append("bad_is_transaction")
    for k, allowed in ENUMS.items():
        v = obj.get(k)
        if v is not None and (not isinstance(v, str) or v not in allowed):
            errs.append(f"bad_{k}")
    for k in ("amount", "balance"):
        v = obj.get(k)
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            errs.append(f"bad_{k}")
    d = obj.get("date")
    if d is not None and not (isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)):
        errs.append("bad_date")
    a = obj.get("account_last4")
    if a is not None and not (isinstance(a, str) and re.fullmatch(r"\d{4}", a)):
        errs.append("bad_account_last4")
    return errs


# ---------------------------------------------------------------------------
# Normalisation & field comparison (lenient on formatting, strict on meaning)
# ---------------------------------------------------------------------------
def _num(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).translate(_DEV)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    return float(m.group(0).replace(",", "")) if m else None


def _norm_str(v):
    if v is None:
        return None
    s = unicodedata.normalize("NFC", str(v)).translate(_DEV).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _norm_alnum(v):
    s = _norm_str(v)
    return re.sub(r"[^0-9a-z@.]", "", s) if s else None


def norm_field(k, v):
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a")):
        return None
    if k in ("amount", "balance"):
        return _num(v)
    if k == "account_last4":
        d = re.sub(r"\D", "", str(v).translate(_DEV))
        return d[-4:] if d else None
    if k in ("instrument", "currency"):
        return _norm_str(v).upper() if _norm_str(v) else None
    if k in ("type", "status"):
        return _norm_str(v)
    if k in ("reference", "counterparty"):
        return _norm_alnum(v)
    if k == "date":
        return _norm_str(v)
    if k == "is_transaction":
        if isinstance(v, bool):
            return v
        return {"true": True, "false": False}.get(str(v).lower())
    return _norm_str(v)


def field_equal(k, pred, gold) -> bool:
    p, g = norm_field(k, pred), norm_field(k, gold)
    if g is None or p is None:
        return g is None and p is None
    if k in ("amount", "balance"):
        return abs(p - g) < 0.005
    return p == g


# ---------------------------------------------------------------------------
# Per-example and aggregate scoring
# ---------------------------------------------------------------------------
def score_example(raw_output: str, gold: dict) -> dict:
    obj = extract_json(raw_output)
    errs = schema_errors(obj)
    pred = obj or {}
    res = {"json_parsed": obj is not None, "schema_valid": obj is not None and not errs,
           "schema_errors": errs}
    correct = {}
    for k in FIELDS:
        correct[k] = field_equal(k, pred.get(k), gold.get(k))
    res["field_correct"] = correct
    res["exact_match"] = obj is not None and all(correct.values())
    # hallucination: predicted a value where gold is null; omission: the reverse
    hall = omit = gold_null = gold_val = 0
    for k in VALUE_FIELDS:
        g = norm_field(k, gold.get(k))
        p = norm_field(k, pred.get(k))
        if g is None:
            gold_null += 1
            hall += p is not None
        else:
            gold_val += 1
            omit += p is None
    res.update(hallucinated=hall, gold_null=gold_null, omitted=omit, gold_valued=gold_val)
    res["pred_is_txn"] = norm_field("is_transaction", pred.get("is_transaction"))
    res["gold_is_txn"] = gold["is_transaction"]
    return res


def aggregate(scored: list[dict]) -> dict:
    n = len(scored)
    if n == 0:
        return {}
    out = {
        "n": n,
        "json_valid": sum(s["json_parsed"] for s in scored) / n,
        "schema_valid": sum(s["schema_valid"] for s in scored) / n,
        "exact_match": sum(s["exact_match"] for s in scored) / n,
    }
    # micro field accuracy over value fields on transaction examples
    txn = [s for s in scored if s["gold_is_txn"]]
    if txn:
        out["field_acc_txn"] = sum(s["field_correct"][k] for s in txn for k in VALUE_FIELDS) / (
            len(txn) * len(VALUE_FIELDS))
        for k in VALUE_FIELDS:
            out[f"acc_{k}"] = sum(s["field_correct"][k] for s in txn) / len(txn)
    gn = sum(s["gold_null"] for s in scored)
    gv = sum(s["gold_valued"] for s in scored)
    out["hallucination_rate"] = sum(s["hallucinated"] for s in scored) / gn if gn else float("nan")
    out["omission_rate"] = sum(s["omitted"] for s in scored) / gv if gv else float("nan")
    # transaction detection (positive class = NOT a transaction, i.e. catching hard negatives)
    tp = sum(1 for s in scored if s["gold_is_txn"] is False and s["pred_is_txn"] is False)
    fp = sum(1 for s in scored if s["gold_is_txn"] is True and s["pred_is_txn"] is False)
    fn = sum(1 for s in scored if s["gold_is_txn"] is False and s["pred_is_txn"] is not False)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    out["nontxn_precision"], out["nontxn_recall"] = prec, rec
    out["nontxn_f1"] = 2 * prec * rec / (prec + rec) if prec and rec and not math.isnan(prec + rec) else 0.0
    out["is_txn_acc"] = sum(1 for s in scored if s["pred_is_txn"] == s["gold_is_txn"]) / n
    return out


def breakdown(rows: list[dict], scored: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r, s in zip(rows, scored):
        vals = r.get(key)
        for v in (vals if isinstance(vals, list) else [vals]):
            groups[v].append(s)
    return {g: aggregate(v) for g, v in sorted(groups.items(), key=lambda x: str(x[0]))}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Paired exact McNemar test (a = system A correct, b = system B correct)."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"b_only": n01, "a_only": n10, "p_value": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b_only": n01, "a_only": n10, "p_value": min(1.0, 2 * p)}


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
