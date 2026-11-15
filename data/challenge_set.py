"""Hand-written challenge set (human-labelled, NOT produced by the generator).

Styles, wording and orderings differ from every generator template, so this measures
robustness on realistic, messy messages. Bank names are fictional.
Writes data/challenge_set.jsonl
"""
import json
import os

N = None


def G(txn=True, type=N, amount=N, currency=N, acct=N, inst=N, cp=N, date=N, ref=N, bal=N, status=N):
    if not txn:
        return {"is_transaction": False, "type": N, "amount": N, "currency": N, "account_last4": N,
                "instrument": N, "counterparty": N, "date": N, "reference": N, "balance": N, "status": N}
    return {"is_transaction": True, "type": type, "amount": amount, "currency": currency,
            "account_last4": acct, "instrument": inst, "counterparty": cp, "date": date,
            "reference": ref, "balance": bal, "status": status}


CHALLENGE = [
    # ---------------- English, unusual orderings & formats ----------------
    ("en", ["reordered"], "UPI/P2M/402917365512/BLINKIT -- INR 349.00 Dr. A/c **2231 dt 03-11-26. Avl Bal INR 8,412.09 -Banyan Bank",
     G(type="debit", amount=349.0, currency="INR", acct="2231", inst="UPI", cp="BLINKIT", date="2026-11-03", ref="402917365512", bal=8412.09, status="success")),
    ("en", ["cr_dr_abbrev"], "A/c XX0091 Cr by Rs 1,50,000.00 on 01Oct26 NEFT-SUNRISE SOFTWARE LLP. Bal Rs 2,04,551.30",
     G(type="credit", amount=150000.0, currency="INR", acct="0091", inst="NEFT", cp="SUNRISE SOFTWARE LLP", date="2026-10-01", bal=204551.3, status="success")),
    ("en", ["lakh_grouping", "no_currency_symbol_on_balance"], "Rs.12,34,567.89 transferred via RTGS to GANGA TEXTILES LTD from a/c ending 7788 on 15/03/2026. UTR SBINR52026031500421",
     G(type="debit", amount=1234567.89, currency="INR", acct="7788", inst="RTGS", cp="GANGA TEXTILES LTD", date="2026-03-15", ref="SBINR52026031500421", status="success")),
    ("en", ["limit_distractor", "two_amounts"], "Kestrel Bank CC xx4410: Rs.2,999 spent at CROMA on 11-11-26. Avl Lmt Rs.47,001. Total Lmt Rs.50,000.",
     G(type="debit", amount=2999.0, currency="INR", acct="4410", inst="CARD", cp="CROMA", date="2026-11-11", status="success")),
    ("en", ["foreign_currency"], "Alert: Card ending 5521 charged EUR 89.90 at BOOKING HOTEL AMSTERDAM on 22 Aug 2026. Not you? Block via app.",
     G(type="debit", amount=89.9, currency="EUR", acct="5521", inst="CARD", cp="BOOKING HOTEL AMSTERDAM", date="2026-08-22", status="success")),
    ("en", ["reversal"], "Rs 1,250 debited from your a/c XX3390 on 05-09-26 for UPI txn to IRCTC has been reversed. UPI Ref 512309871234.",
     G(type="credit", amount=1250.0, currency="INR", acct="3390", inst="UPI", cp="IRCTC", date="2026-09-05", ref="512309871234", status="reversed")),
    ("en", ["failed", "no_date"], "FAILED: UPI payment of Rs.540 to ZEPTO. Money will be auto-refunded within 48 hrs if deducted. Ref 778812340011",
     G(type="debit", amount=540.0, currency="INR", inst="UPI", cp="ZEPTO", ref="778812340011", status="failed")),
    ("en", ["all_caps", "missing_space"], "RS.75000.00CREDITED TO A/C XXXX6120 BY IMPS FROM ANJALI MISHRA ON 18-07-2026.REF 619900123456.AVL BAL RS.81200.00",
     G(type="credit", amount=75000.0, currency="INR", acct="6120", inst="IMPS", cp="ANJALI MISHRA", date="2026-07-18", ref="619900123456", bal=81200.0, status="success")),
    ("en", ["atm", "location_text"], "Cash Rs 10,000 withdrawn at ATM MG ROAD from A/c *8845 on 02/02/2026 13:44. Bal Rs 3,215.",
     G(type="debit", amount=10000.0, currency="INR", acct="8845", inst="ATM", date="2026-02-02", bal=3215.0, status="success")),
    ("en", ["nach", "emi"], "NACH debit of Rs.18,450 for HOME LOAN EMI processed on 05-12-2025 from a/c XX0702. Bal Rs.6,010.55",
     G(type="debit", amount=18450.0, currency="INR", acct="0702", inst="NACH", cp="HOME LOAN EMI", date="2025-12-05", bal=6010.55, status="success")),
    ("en", ["vpa_counterparty"], "You paid Rs 60 to chaiwala.raju7@neemupi from A/c XX1190 on 09Nov26 via UPI (Ref 600123998877).",
     G(type="debit", amount=60.0, currency="INR", acct="1190", inst="UPI", cp="chaiwala.raju7@neemupi", date="2026-11-09", ref="600123998877", status="success")),
    ("en", ["truncated"], "Dear Customer, Rs.4,200.00 has been debited from your account XX5566 towards",
     G(type="debit", amount=4200.0, currency="INR", acct="5566", status="success")),
    ("en", ["decimal_no_grouping", "paise"], "INR 0.50 credited to a/c x9910 as cashback from BLINKIT on 2026-10-30.",
     G(type="credit", amount=0.5, currency="INR", acct="9910", cp="BLINKIT", date="2026-10-30", status="success")),
    # ---------------- English hard negatives ----------------
    ("en", ["hard_negative", "otp_with_amount"], "Do not share. 482913 is OTP for txn of INR 12,500.00 at AMAZON on card XX4410. -Kestrel Bank",
     G(txn=False)),
    ("en", ["hard_negative", "future_debit"], "Rs 499 will be auto-debited on 20-11-26 for your NETFLIX mandate from a/c XX1190. Ensure sufficient balance.",
     G(txn=False)),
    ("en", ["hard_negative", "bill_due"], "Statement for card xx4410: Total due Rs.12,840, Min due Rs.642, pay by 25/11/2026.",
     G(txn=False)),
    ("en", ["hard_negative", "balance_only"], "Your available balance in A/c XX5566 is Rs. 9,876.54 as of 14-11-2026 10:00.",
     G(txn=False)),
    ("en", ["hard_negative", "request_money"], "PRIYA SHARMA has requested Rs.1,500 from you on UPI. Approve only if you know the requester.",
     G(txn=False)),
    ("en", ["hard_negative", "promo"], "Flat Rs 250 off on your first DMART order above Rs 1,500! Use code DM250. Valid till 30 Nov.",
     G(txn=False)),
    # ---------------- Hindi ----------------
    ("hi", ["reordered"], "बनयान बैंक: UPI रेफ़ 300198765432 | ₹1,299.00 खाता XX4321 से डेबिट | प्राप्तकर्ता MYNTRA | 12.11.2026",
     G(type="debit", amount=1299.0, currency="INR", acct="4321", inst="UPI", cp="MYNTRA", date="2026-11-12", ref="300198765432", status="success")),
    ("hi", ["dev_digits"], "आपके खाते XX७७१२ में ₹२५,०००.०० NEFT द्वारा VEDA HOSPITALS से जमा हुए, दिनांक ०१-११-२०२६। शेष ₹३१,४०२.१०।",
     G(type="credit", amount=25000.0, currency="INR", acct="7712", inst="NEFT", cp="VEDA HOSPITALS", date="2026-11-01", bal=31402.1, status="success")),
    ("hi", ["failed"], "लेनदेन विफल: ZOMATO को ₹420 का UPI भुगतान नहीं हुआ। कटी हुई राशि 2 दिन में वापस आ जाएगी। रेफ़ 556677889900",
     G(type="debit", amount=420.0, currency="INR", inst="UPI", cp="ZOMATO", ref="556677889900", status="failed")),
    ("hi", ["limit_distractor"], "क्रेडिट कार्ड XX4410 से AMAZON पर ₹5,600 का भुगतान हुआ, 03-10-26। बची हुई लिमिट ₹44,400।",
     G(type="debit", amount=5600.0, currency="INR", acct="4410", inst="CARD", cp="AMAZON", date="2026-10-03", status="success")),
    ("hi", ["atm"], "एटीएम से ₹2,000 की निकासी, खाता *8845, 07/06/2026। उपलब्ध शेष ₹1,215।",
     G(type="debit", amount=2000.0, currency="INR", acct="8845", inst="ATM", date="2026-06-07", bal=1215.0, status="success")),
    ("hi", ["reversal"], "UPI लेनदेन (रेफ़ 401122334455) में कटे ₹999 आपके खाते XX3390 में वापस जमा कर दिए गए हैं।",
     G(type="credit", amount=999.0, currency="INR", acct="3390", inst="UPI", ref="401122334455", status="reversed")),
    ("hi", ["counterparty_person"], "SURESH YADAV ने आपके खाते XX2208 में IMPS से ₹3,500 भेजे हैं। तारीख 21-08-2026।",
     G(type="credit", amount=3500.0, currency="INR", acct="2208", inst="IMPS", cp="SURESH YADAV", date="2026-08-21", status="success")),
    ("hi", ["hard_negative", "otp_with_amount"], "₹8,999 के भुगतान के लिए आपका OTP 771204 है। किसी से साझा न करें। - मॉनसून बैंक",
     G(txn=False)),
    ("hi", ["hard_negative", "future_debit"], "आपकी LIC PREMIUM की ₹12,000 की किस्त 28-11-2026 को खाता XX0702 से कटेगी।",
     G(txn=False)),
    ("hi", ["hard_negative", "balance_only"], "प्रिय ग्राहक, आपके खाते XX5566 का शेष ₹४,५६७.८९ है।",
     G(txn=False)),
    ("hi", ["hard_negative", "promo"], "त्योहारी ऑफ़र! ₹10,000 से ज़्यादा की खरीदारी पर ₹1,000 का कैशबैक। ऑफ़र 31-10-2026 तक।",
     G(txn=False)),
    ("hi", ["truncated"], "आपके खाते XX1190 से ₹15,000 डेबिट किए गए और",
     G(type="debit", amount=15000.0, currency="INR", acct="1190", status="success")),
    # ---------------- Hinglish ----------------
    ("hinglish", ["reordered"], "Banyan Bank alert - ZOMATO ko Rs 312.50 UPI se pay hua | A/c xx4321 | 13-11-26 | Ref 300554433221",
     G(type="debit", amount=312.5, currency="INR", acct="4321", inst="UPI", cp="ZOMATO", date="2026-11-13", ref="300554433221", status="success")),
    ("hinglish", ["salary", "lakh_grouping"], "Badhai ho! Salary Rs 1,12,500 aapke A/c XX0091 mein NEFT se credit hui (NORTHSTAR LOGISTICS), 31-10-2026. Bal Rs 1,48,220",
     G(type="credit", amount=112500.0, currency="INR", acct="0091", inst="NEFT", cp="NORTHSTAR LOGISTICS", date="2026-10-31", bal=148220.0, status="success")),
    ("hinglish", ["lowercase", "missing_space"], "a/c xx8845 se rs.500debit hue atm se 09-09-26 ko. bal rs.715",
     G(type="debit", amount=500.0, currency="INR", acct="8845", inst="ATM", date="2026-09-09", bal=715.0, status="success")),
    ("hinglish", ["failed"], "Payment fail! Rs 2,100 OLA ko nahi gaya. Paisa kata ho to 3 din mein wapas. UPI Ref 990011223344",
     G(type="debit", amount=2100.0, currency="INR", inst="UPI", cp="OLA", ref="990011223344", status="failed")),
    ("hinglish", ["foreign_currency"], "Card XX5521 se USD 12.99 ka payment GOOGLE ONE ko hua, 01-11-2026.",
     G(type="debit", amount=12.99, currency="USD", acct="5521", inst="CARD", cp="GOOGLE ONE", date="2026-11-01", status="success")),
    ("hinglish", ["refund"], "Refund aa gaya! FLIPKART se Rs.1,849 aapke A/c XX2208 mein credit, 06-11-26.",
     G(type="credit", amount=1849.0, currency="INR", acct="2208", cp="FLIPKART", date="2026-11-06", status="success")),
    ("hinglish", ["autopay"], "SPOTIFY ke Rs 119 AutoPay se kat gaye A/c XX1190 se, 10-11-2026. Ref AP77120934",
     G(type="debit", amount=119.0, currency="INR", acct="1190", inst="AUTOPAY", cp="SPOTIFY", date="2026-11-10", ref="AP77120934", status="success")),
    ("hinglish", ["hard_negative", "otp_with_amount"], "Rs 4,500 ke txn ke liye OTP 339201 hai. Kisi ko mat batana. -Neem Bank",
     G(txn=False)),
    ("hinglish", ["hard_negative", "request_money"], "ROHIT DAS ne aapse Rs 800 maange hain UPI par. Jaante ho tabhi approve karein.",
     G(txn=False)),
    ("hinglish", ["hard_negative", "future_debit"], "Kal 15-11-26 ko aapke A/c XX0702 se Rs 2,300 ki EMI kategi. Balance maintain rakhein.",
     G(txn=False)),
    ("hinglish", ["hard_negative", "bill_due"], "JIO FIBER ka Rs 1,178 ka bill 18-11-2026 tak bharna hai. Abhi pay karein.",
     G(txn=False)),
    ("hinglish", ["code_mixed_script"], "आपके A/c XX4321 se Rs 650 debit hue SWIGGY ko UPI se, 14-11-26. Ref 300777888999",
     G(type="debit", amount=650.0, currency="INR", acct="4321", inst="UPI", cp="SWIGGY", date="2026-11-14", ref="300777888999", status="success")),
    ("hinglish", ["truncated"], "Rs 9,999 aapke Card XX4410 se CROMA par",
     G(type="debit", amount=9999.0, currency="INR", acct="4410", inst="CARD", cp="CROMA", status="success")),
]


def write():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "challenge_set.jsonl"), "w", encoding="utf-8") as f:
        for i, (lang, tags, sms, gold) in enumerate(CHALLENGE):
            f.write(json.dumps({"id": f"challenge-{i:03d}", "family": "challenge", "lang": lang,
                                "template_id": "handwritten", "sms": sms, "gold": gold,
                                "tags": tags, "split": "challenge"}, ensure_ascii=False) + "\n")
    return len(CHALLENGE)


if __name__ == "__main__":
    print(write())
