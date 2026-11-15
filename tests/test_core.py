"""Model-free checks for the generator and scoring. Run: python tests/test_core.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from smsgen import FIELDS, build_dataset, gold_to_str  # noqa: E402
import regex_baseline as RB  # noqa: E402
import scoring as S  # noqa: E402

DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def test_splits_and_leakage():
    ds = build_dataset()
    assert [len(ds[k]) for k in ["train", "dev", "test_id", "test_ood"]] == [2400, 200, 400, 400]
    train_sms = {e.sms for e in ds["train"]}
    for k in ["dev", "test_id", "test_ood"]:
        assert not train_sms & {e.sms for e in ds[k]}
    assert not {e.template_id for e in ds["test_ood"]} & {e.template_id for e in ds["train"]}


def test_gold_is_grounded_in_text():
    import re
    ds = build_dataset()
    for split in ds.values():
        for e in split:
            g, t = e.gold, e.sms.translate(DEV)
            nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", t)]
            if g["amount"] is not None:
                assert any(abs(n - g["amount"]) < 0.005 for n in nums), e.sms
            if g["account_last4"]:
                assert g["account_last4"] in t, e.sms
            if g["reference"]:
                assert g["reference"].upper() in t.upper(), e.sms
            if not g["is_transaction"]:
                assert all(g[k] is None for k in FIELDS[1:])


def test_scoring():
    ds = build_dataset(n_train=10, n_dev=0, n_test_id=50, n_test_ood=0)
    for e in ds["test_id"]:
        s = S.score_example("```json\n" + gold_to_str(e.gold) + "\n```", e.gold)
        assert s["exact_match"] and s["schema_valid"] and s["hallucinated"] == 0
    g = ds["test_id"][0].gold
    assert not S.score_example("no json here", g)["json_parsed"]
    bad = dict(g, balance=123.0) if g["balance"] is None else dict(g, balance=None)
    s = S.score_example(json.dumps(bad), g)
    assert not s["exact_match"] and (s["hallucinated"] == 1 or s["omitted"] == 1)


def test_challenge_set_valid():
    for line in open(os.path.join(ROOT, "data", "challenge_set.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        assert S.schema_errors(r["gold"]) == [], r["sms"]
        RB.predict(r["sms"])   # baseline never crashes


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
