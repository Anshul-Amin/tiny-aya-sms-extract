"""Synthetic Indian bank / UPI SMS generator with exact gold labels.

Design
------
* Every template is a list of SEGMENTS joined by '‖'. The first segment is the core
  (always kept); later segments can be dropped (optional '?' or by truncation). A field
  is in the gold label ONLY if its slot survived -> the gold never contains information
  that is not in the text (this is what lets us measure hallucination honestly).
* '{@inst}' reveals the instrument without printing it (e.g. "UPI Ref" text implies UPI).
* The LAST template of each (family, language) is held out -> `test_ood` uses message
  formats never seen in training.
* Bank names are FICTIONAL and messages contain no links or phone numbers, so the data
  cannot be confused with, or reused as, real bank messages.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

FIELDS = ["is_transaction", "type", "amount", "currency", "account_last4", "instrument",
          "counterparty", "date", "reference", "balance", "status"]
INSTRUMENTS = ["UPI", "CARD", "NEFT", "IMPS", "RTGS", "ATM", "NACH", "AUTOPAY"]
LANGS = ["en", "hi", "hinglish"]

FICTIONAL_BANKS = ["Kestrel Bank", "Banyan Bank", "Peacock Bank", "Monsoon Bank",
                   "Neem Bank", "Saffron Bank", "Lotus Demo Bank", "Indigo Coop Bank"]
VPA_HANDLES = ["@kestrel", "@banyan", "@peacockpay", "@monsoon", "@neemupi", "@saffron"]

MERCHANTS = ["SWIGGY", "ZOMATO", "AMAZON", "FLIPKART", "BIGBASKET", "IRCTC", "UBER", "OLA",
             "MYNTRA", "DMART", "RELIANCE FRESH", "APOLLO PHARMACY", "BOOKMYSHOW",
             "SHARMA GENERAL STORE", "GUPTA SWEETS", "CAFE COFFEE CORNER", "METRO CASH CARRY",
             "HP PETROL PUMP", "DECATHLON", "NYKAA", "CROMA", "BLINKIT", "ZEPTO"]
PERSONS = ["RAMESH KUMAR", "PRIYA SHARMA", "ANIL VERMA", "SUNITA DEVI", "ARJUN NAIR",
           "FATIMA SHAIKH", "GURPREET SINGH", "MEERA IYER", "ROHIT DAS", "KAVYA REDDY",
           "IMRAN KHAN", "POOJA PATEL", "VIKRAM JOSHI", "ANJALI MISHRA", "SURESH YADAV"]
EMPLOYERS = ["ACME TECHNOLOGIES PVT LTD", "SUNRISE SOFTWARE LLP", "GANGA TEXTILES LTD",
             "NORTHSTAR LOGISTICS", "BLUEWAVE SERVICES PVT LTD", "VEDA HOSPITALS"]
SUBSCRIPTIONS = ["NETFLIX", "SPOTIFY", "YOUTUBE PREMIUM", "JIO FIBER", "AIRTEL POSTPAID",
                 "TATA PLAY", "GOOGLE ONE", "HOTSTAR", "LIC PREMIUM", "SIP MUTUAL FUND"]
BILLERS = ["CITY POWER LTD", "JAL BOARD", "METRO GAS", "AIRTEL POSTPAID", "JIO FIBER",
           "MUNICIPAL TAX"]
FOREIGN = [("USD", "$"), ("EUR", "€"), ("GBP", "£"), ("AED", "AED ")]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Templates: family -> {"meta": {...}, "en": [...], "hi": [...], "hinglish": [...]}
# meta: is_txn, type, status (per-template override via tuple (text, status)),
#       inst (choices for {inst}/{@inst}), cp (counterparty pool name)
# ---------------------------------------------------------------------------
T = {
    "upi_debit": {
        "meta": dict(is_txn=True, type="debit", status="success", inst=["UPI"], cp="upi_party"),
        "en": [
            "Your A/c {acct} is debited for {amt}‖ on {date}‖ towards {cp}‖ via {inst}‖. UPI Ref No {ref}‖. Avl Bal {bal}",
            "{bank}: {amt} debited from a/c {acct}‖ on {date}‖ to VPA {cp}{@inst}‖ (UPI Ref {ref})‖. Avl bal {bal}",
            "Sent {amt} from {bank} A/c {acct} to {cp} on {date}{@inst}‖. Ref: {ref}‖. Bal: {bal}",
        ],
        "hi": [
            "आपके खाते {acct} से {amt} डेबिट किए गए‖ दिनांक {date}‖, {cp} को‖ {inst} द्वारा‖। UPI संदर्भ {ref}‖। उपलब्ध शेष {bal}",
            "{bank}: खाता {acct} से {amt} {cp} को भेजे गए{@inst}‖ ({date})‖। संदर्भ संख्या {ref}‖। शेष राशि {bal}",
            "{amt} का UPI भुगतान{@inst} {cp} को सफल रहा, खाता {acct}‖, दिनांक {date}‖। रेफ़ {ref}",
        ],
        "hinglish": [
            "Aapke A/c {acct} se {amt} debit hue‖ {date} ko‖ {cp} ko‖ {inst} se‖. UPI Ref {ref}‖. Avl bal {bal}",
            "{bank}: A/c {acct} se {amt} {cp} ko bheje gaye{@inst}‖ ({date})‖. Ref no {ref}‖. Bal {bal}",
            "{amt} ka UPI payment{@inst} {cp} ko successful, A/c {acct}‖, date {date}‖. Ref {ref}",
        ],
    },
    "upi_credit": {
        "meta": dict(is_txn=True, type="credit", status="success", inst=["UPI"], cp="upi_party"),
        "en": [
            "Dear Customer, your A/c {acct} is credited with {amt}‖ on {date}‖ by {cp}‖ via {inst}‖. Ref {ref}‖. Avl Bal {bal}",
            "{amt} received in {bank} a/c {acct} from {cp}{@inst}‖ on {date}‖. UPI Ref:{ref}",
            "Money received! {cp} sent you {amt} via {inst}‖ on {date}‖. Credited to A/c {acct}‖. Bal {bal}",
        ],
        "hi": [
            "आपके खाते {acct} में {amt} जमा किए गए‖ दिनांक {date}‖, {cp} द्वारा‖ {inst} से‖। संदर्भ {ref}‖। उपलब्ध शेष {bal}",
            "{cp} ने आपको {amt} भेजे{@inst}‖, खाता {acct} में जमा‖ ({date})‖। UPI रेफ़ {ref}",
            "खाता {acct} में {cp} से {amt} प्राप्त हुए‖ {date} को‖ {inst} द्वारा‖। शेष {bal}",
        ],
        "hinglish": [
            "Aapke A/c {acct} mein {amt} credit hue‖ {date} ko‖ {cp} se‖ {inst} dwara‖. Ref {ref}‖. Avl bal {bal}",
            "{cp} ne aapko {amt} bheje{@inst}‖, A/c {acct} mein jama‖ ({date})‖. UPI Ref {ref}",
            "A/c {acct} mein {cp} se {amt} mile‖ {date} ko‖ {inst} se‖. Bal {bal}",
        ],
    },
    "card_spend": {
        "meta": dict(is_txn=True, type="debit", status="success", inst=["CARD"], cp="merchant", foreign=0.25),
        "en": [
            "You have spent {amt} on your {bank} Credit Card {card}‖ at {cp}‖ on {date}‖. Avl limit: {limit}",
            "Txn of {amt} on Card {card} at {cp}‖ on {date}‖. Ref {ref}‖. Available credit limit {limit}",
            "{bank} Debit Card {card} used for {amt} at {cp}‖ on {date} {time}‖. Avl bal {bal}",
        ],
        "hi": [
            "आपके {bank} क्रेडिट कार्ड {card} से {cp} पर {amt} खर्च किए गए‖ दिनांक {date}‖। उपलब्ध लिमिट {limit}",
            "कार्ड {card} से {amt} का लेनदेन {cp} पर‖, {date}‖। संदर्भ {ref}‖। उपलब्ध क्रेडिट लिमिट {limit}",
            "{bank} डेबिट कार्ड {card} का उपयोग {cp} पर {amt} के लिए किया गया‖, {date} {time}‖। शेष {bal}",
        ],
        "hinglish": [
            "Aapke {bank} Credit Card {card} se {cp} par {amt} kharch hue‖ {date} ko‖. Avl limit {limit}",
            "Card {card} se {amt} ka txn {cp} par‖, {date}‖. Ref {ref}‖. Avl credit limit {limit}",
            "{bank} Debit Card {card} use hua {cp} par {amt} ke liye‖, {date} {time}‖. Bal {bal}",
        ],
    },
    "atm_withdrawal": {
        "meta": dict(is_txn=True, type="debit", status="success", inst=["ATM"], cp=None),
        "en": [
            "{amt} withdrawn from A/c {acct}‖ at {inst}‖ on {date}‖. Avl Bal {bal}",
            "Cash withdrawal{@inst} of {amt} from your {bank} A/c {acct}‖ on {date} {time}‖. Bal: {bal}",
            "ATM txn{@inst}: {amt} debited from a/c {acct}‖ on {date}‖. Avl bal {bal}",
        ],
        "hi": [
            "खाता {acct} से {amt} की नकद निकासी{@inst}‖ {date} को‖। उपलब्ध शेष {bal}",
            "{inst} से {amt} निकाले गए, खाता {acct}‖, दिनांक {date}‖। शेष राशि {bal}",
            "{bank} ATM लेनदेन{@inst}: खाता {acct} से {amt} डेबिट‖, {date} {time}‖। शेष {bal}",
        ],
        "hinglish": [
            "A/c {acct} se {amt} cash nikale gaye{@inst}‖ {date} ko‖. Avl bal {bal}",
            "{inst} se {amt} withdraw hue, A/c {acct}‖, date {date}‖. Bal {bal}",
            "{bank} ATM txn{@inst}: A/c {acct} se {amt} debit‖, {date} {time}‖. Bal {bal}",
        ],
    },
    "transfer_credit": {
        "meta": dict(is_txn=True, type="credit", status="success", inst=["NEFT", "IMPS", "RTGS"], cp="transfer_party"),
        "en": [
            "Your A/c {acct} has been credited with {amt}‖ on {date}‖ by {inst}‖ from {cp}‖. UTR {ref}‖. Avl Bal {bal}",
            "{inst} credit of {amt} received in A/c {acct} from {cp}‖ on {date}‖. Ref {ref}",
            "Salary credited: {amt} to A/c {acct}‖ from {cp}‖ via {inst}‖ on {date}‖. Bal {bal}",
        ],
        "hi": [
            "आपके खाते {acct} में {inst} द्वारा {amt} जमा हुए‖, {cp} से‖, दिनांक {date}‖। UTR {ref}‖। उपलब्ध शेष {bal}",
            "{cp} से {amt} का {inst} क्रेडिट खाता {acct} में प्राप्त‖ ({date})‖। संदर्भ {ref}",
            "वेतन जमा: खाता {acct} में {amt}‖, {cp} द्वारा‖ {inst} से‖, {date}‖। शेष {bal}",
        ],
        "hinglish": [
            "Aapke A/c {acct} mein {inst} se {amt} credit hue‖ {cp} se‖, {date} ko‖. UTR {ref}‖. Avl bal {bal}",
            "{cp} se {amt} ka {inst} credit A/c {acct} mein aaya‖ ({date})‖. Ref {ref}",
            "Salary aa gayi: A/c {acct} mein {amt}‖ {cp} se‖ {inst} dwara‖, {date}‖. Bal {bal}",
        ],
    },
    "transfer_debit": {
        "meta": dict(is_txn=True, type="debit", status="success", inst=["NEFT", "IMPS", "RTGS"], cp="person"),
        "en": [
            "{amt} debited from A/c {acct}‖ on {date}‖ for {inst} transfer‖ to {cp}‖. Ref {ref}‖. Avl Bal {bal}",
            "Your {inst} of {amt} to {cp} from A/c {acct} is successful‖. Ref No {ref}‖. Date {date}",
            "A/c {acct} debited by {amt}‖ on {date}‖; {inst} to {cp}‖, UTR {ref}",
        ],
        "hi": [
            "खाता {acct} से {amt} डेबिट‖, {date}‖, {inst} ट्रांसफ़र‖ {cp} को‖। संदर्भ {ref}‖। उपलब्ध शेष {bal}",
            "{cp} को {amt} का आपका {inst} ट्रांसफ़र सफल रहा, खाता {acct}‖। संदर्भ संख्या {ref}‖। दिनांक {date}",
            "खाता {acct} से {amt} कटे‖ {date} को‖; {cp} को {inst}‖, UTR {ref}",
        ],
        "hinglish": [
            "A/c {acct} se {amt} debit‖, {date}‖, {inst} transfer‖ {cp} ko‖. Ref {ref}‖. Avl bal {bal}",
            "{cp} ko {amt} ka aapka {inst} transfer successful, A/c {acct}‖. Ref no {ref}‖. Date {date}",
            "A/c {acct} se {amt} kat gaye‖ {date} ko‖; {cp} ko {inst}‖, UTR {ref}",
        ],
    },
    "refund_reversal": {
        "meta": dict(is_txn=True, type="credit", status="success", inst=[], cp="merchant"),
        "en": [
            ("Refund of {amt} from {cp} credited to your A/c {acct}‖ on {date}‖. Avl Bal {bal}", "success"),
            ("{amt} debited on {date} for txn to {cp} has been reversed to your A/c {acct}‖. Ref {ref}", "reversed"),
            ("Your refund of {amt} for order at {cp} is processed to Card {card}‖ on {date}", "success"),
        ],
        "hi": [
            ("{cp} से {amt} का रिफ़ंड आपके खाते {acct} में जमा किया गया‖, दिनांक {date}‖। उपलब्ध शेष {bal}", "success"),
            ("{date} को {cp} के लेनदेन में कटे {amt} आपके खाते {acct} में वापस कर दिए गए हैं‖। संदर्भ {ref}", "reversed"),
            ("{cp} पर ऑर्डर का {amt} रिफ़ंड कार्ड {card} में भेज दिया गया है‖, {date}", "success"),
        ],
        "hinglish": [
            ("{cp} se {amt} ka refund aapke A/c {acct} mein credit hua‖ {date} ko‖. Avl bal {bal}", "success"),
            ("{date} ko {cp} txn mein kate {amt} aapke A/c {acct} mein wapas aa gaye‖. Ref {ref}", "reversed"),
            ("{cp} order ka {amt} refund Card {card} mein bhej diya gaya‖, {date}", "success"),
        ],
    },
    "failed_txn": {
        "meta": dict(is_txn=True, type="debit", status="failed", inst=["UPI"], cp="merchant"),
        "en": [
            "Txn of {amt} to {cp} via {inst} FAILED‖ on {date}‖. Amount, if debited, will be refunded‖. Ref {ref}",
            "Your payment of {amt} to {cp} could not be processed{@inst}‖ (UPI Ref {ref})‖. A/c {acct} not debited",
            "Transaction declined: {amt} at {cp} on Card {card}‖ on {date}‖. Reason: insufficient balance",
        ],
        "hi": [
            "{inst} द्वारा {cp} को {amt} का लेनदेन विफल रहा‖, दिनांक {date}‖। अगर राशि कटी है तो वापस आ जाएगी‖। संदर्भ {ref}",
            "{cp} को {amt} का आपका भुगतान नहीं हो सका{@inst}‖ (UPI रेफ़ {ref})‖। खाता {acct} से राशि नहीं कटी",
            "लेनदेन अस्वीकृत: कार्ड {card} से {cp} पर {amt}‖, {date}‖। कारण: अपर्याप्त शेष",
        ],
        "hinglish": [
            "{inst} se {cp} ko {amt} ka txn fail ho gaya‖ {date} ko‖. Paisa kata hai to wapas aa jayega‖. Ref {ref}",
            "{cp} ko {amt} ka aapka payment nahi ho paya{@inst}‖ (UPI Ref {ref})‖. A/c {acct} se paisa nahi kata",
            "Txn decline hua: Card {card} se {cp} par {amt}‖, {date}‖. Reason: balance kam hai",
        ],
    },
    "autopay_emi": {
        "meta": dict(is_txn=True, type="debit", status="success", inst=["NACH", "AUTOPAY"], cp="subscription"),
        "en": [
            "{amt} debited from A/c {acct} towards EMI for loan {ref}‖ via {inst}‖ on {date}‖. Avl Bal {bal}",
            "AutoPay{@inst=AUTOPAY} of {amt} for {cp} executed from A/c {acct}‖ on {date}‖. Ref {ref}",
            "Your {cp} subscription of {amt} was paid via {inst}‖ from A/c {acct}‖ on {date}",
        ],
        "hi": [
            "लोन {ref} की EMI के लिए खाता {acct} से {amt} डेबिट‖ {inst} द्वारा‖, दिनांक {date}‖। उपलब्ध शेष {bal}",
            "{cp} के लिए {amt} का ऑटोपे{@inst=AUTOPAY} खाता {acct} से पूरा हुआ‖, {date}‖। संदर्भ {ref}",
            "आपकी {cp} सदस्यता के {amt} {inst} से चुकाए गए‖, खाता {acct}‖, {date}",
        ],
        "hinglish": [
            "Loan {ref} ki EMI ke liye A/c {acct} se {amt} debit‖ {inst} se‖, {date} ko‖. Avl bal {bal}",
            "{cp} ke liye {amt} ka AutoPay{@inst=AUTOPAY} A/c {acct} se ho gaya‖, {date}‖. Ref {ref}",
            "Aapki {cp} subscription ke {amt} {inst} se pay hue‖, A/c {acct}‖, {date}",
        ],
    },
    # ---------------- hard negatives: money mentioned, but nothing moved ----------------
    "otp": {
        "meta": dict(is_txn=False, cp="merchant"),
        "en": [
            "{otp} is the OTP for your txn of {amt} at {cp} on {bank} Card {card}. Valid for 5 mins. Do not share it with anyone.",
            "OTP {otp} to authorise payment of {amt} to {cp}. Never share your OTP. -{bank}",
        ],
        "hi": [
            "{otp} आपके {bank} कार्ड {card} से {cp} पर {amt} के लेनदेन का OTP है। 5 मिनट तक मान्य। इसे किसी से साझा न करें।",
            "{cp} को {amt} के भुगतान के लिए OTP {otp} है। अपना OTP कभी किसी को न बताएँ। -{bank}",
        ],
        "hinglish": [
            "{otp} aapke {bank} Card {card} se {cp} par {amt} ke txn ka OTP hai. 5 min tak valid. Kisi ke saath share na karein.",
            "{cp} ko {amt} payment ke liye OTP {otp} hai. Apna OTP kabhi kisi ko na batayein. -{bank}",
        ],
    },
    "promo": {
        "meta": dict(is_txn=False, cp=None),
        "en": [
            "Get cashback up to {amt} on your next UPI payment with {bank}! Offer valid till {date}. T&C apply.",
            "Pre-approved loan of up to {amt} is waiting for you. Apply in the {bank} app today.",
        ],
        "hi": [
            "{bank} के साथ अगले UPI भुगतान पर {amt} तक का कैशबैक पाएँ! ऑफ़र {date} तक मान्य। नियम व शर्तें लागू।",
            "आपके लिए {amt} तक का प्री-अप्रूव्ड लोन तैयार है। आज ही {bank} ऐप में आवेदन करें।",
        ],
        "hinglish": [
            "{bank} ke saath agle UPI payment par {amt} tak cashback paayein! Offer {date} tak valid. T&C apply.",
            "Aapke liye {amt} tak ka pre-approved loan ready hai. Aaj hi {bank} app mein apply karein.",
        ],
    },
    "bill_due": {
        "meta": dict(is_txn=False, cp="biller"),
        "en": [
            "Your {bank} Credit Card {card} bill of {amt} is due on {date}. Minimum due {amt2}. Please pay on time to avoid charges.",
            "Reminder: {cp} bill of {amt} is due on {date}. Ignore if already paid.",
        ],
        "hi": [
            "आपके {bank} क्रेडिट कार्ड {card} का {amt} का बिल {date} को देय है। न्यूनतम देय राशि {amt2}। शुल्क से बचने के लिए समय पर भुगतान करें।",
            "रिमाइंडर: {cp} का {amt} का बिल {date} को देय है। यदि भुगतान कर दिया है तो इसे अनदेखा करें।",
        ],
        "hinglish": [
            "Aapke {bank} Credit Card {card} ka {amt} ka bill {date} ko due hai. Minimum due {amt2}. Charges se bachne ke liye time par pay karein.",
            "Reminder: {cp} ka {amt} ka bill {date} ko due hai. Pay kar diya hai to ignore karein.",
        ],
    },
    "scheduled_debit": {
        "meta": dict(is_txn=False, cp="subscription"),
        "en": [
            "{amt} will be debited from A/c {acct} on {date} towards {cp} AutoPay. Please maintain sufficient balance.",
            "Upcoming EMI of {amt} for loan {ref} will be deducted on {date}. -{bank}",
        ],
        "hi": [
            "{date} को {cp} ऑटोपे के लिए खाता {acct} से {amt} डेबिट किए जाएँगे। कृपया पर्याप्त शेष रखें।",
            "लोन {ref} की {amt} की आगामी EMI {date} को कटेगी। -{bank}",
        ],
        "hinglish": [
            "{date} ko {cp} AutoPay ke liye A/c {acct} se {amt} debit honge. Kripya sufficient balance rakhein.",
            "Loan {ref} ki {amt} ki agli EMI {date} ko kategi. -{bank}",
        ],
    },
    "balance_info": {
        "meta": dict(is_txn=False, cp=None),
        "en": [
            "Avl bal in your {bank} A/c {acct} is {bal} as on {date}.",
            "Dear Customer, the balance in A/c {acct} on {date} is {bal}. -{bank}",
        ],
        "hi": [
            "{date} को आपके {bank} खाते {acct} में उपलब्ध शेष {bal} है।",
            "प्रिय ग्राहक, खाता {acct} में {date} को शेष राशि {bal} है। -{bank}",
        ],
        "hinglish": [
            "{date} ko aapke {bank} A/c {acct} mein avl bal {bal} hai.",
            "Dear Customer, A/c {acct} mein {date} ko balance {bal} hai. -{bank}",
        ],
    },
}
TXN_FAMILIES = [f for f, v in T.items() if v["meta"]["is_txn"]]
NEG_FAMILIES = [f for f, v in T.items() if not v["meta"]["is_txn"]]


# ---------------------------------------------------------------------------
# Value renderers
# ---------------------------------------------------------------------------
def indian_group(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def fmt_number(v: float, rng: random.Random) -> str:
    whole = int(round(v * 100)) // 100
    paise = int(round(v * 100)) % 100
    style = rng.choice(["plain", "indian", "indian", "western"])
    w = str(whole) if style == "plain" else indian_group(whole) if style == "indian" else f"{whole:,}"
    if paise == 0 and rng.random() < 0.5:
        return w
    return f"{w}.{paise:02d}"


INR_FORMATS = {
    "en": ["Rs.{n}", "Rs {n}", "INR {n}", "₹{n}", "Rs.{n}/-", "INR{n}"],
    "hi": ["₹{n}", "रु. {n}", "रुपये {n}", "Rs.{n}", "₹ {n}"],
    "hinglish": ["Rs {n}", "Rs.{n}", "₹{n}", "{n} rupaye", "INR {n}"],
}


def rand_amount(rng: random.Random, lo=1.0, hi=500000.0) -> float:
    import math
    v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return round(v, 2) if rng.random() < 0.6 else float(int(v))


def fmt_inr(v: float, lang: str, rng: random.Random) -> str:
    return rng.choice(INR_FORMATS[lang]).format(n=fmt_number(v, rng))


def fmt_foreign(v: float, code: str, sym: str, rng: random.Random) -> str:
    n = f"{v:.2f}"
    return rng.choice([f"{code} {n}", f"{sym}{n}" if sym.strip() != code else f"{code} {n}", f"{code}{n}"])


def fmt_date(d: date, rng: random.Random) -> str:
    style = rng.randrange(8)
    m = MONTHS[d.month - 1]
    return [
        f"{d.day:02d}-{d.month:02d}-{d.year % 100:02d}",
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.day:02d}{m}{d.year % 100:02d}",
        f"{d.day:02d}-{m}-{d.year}",
        f"{d.year}-{d.month:02d}-{d.day:02d}",
        f"{d.day:02d}.{d.month:02d}.{d.year}",
        f"{d.day} {m} {d.year}",
        f"{d.day:02d}-{m.upper()}-{d.year % 100:02d}",
    ][style]


def fmt_acct(l4: str, lang: str, rng: random.Random) -> str:
    opts = [f"XX{l4}", f"XXXXXX{l4}", f"*{l4}", f"x{l4}", f"XXXX{l4}", f"**{l4}"]
    if lang == "en":
        opts += [f"no. XX{l4}", f"ending {l4}"]
    return rng.choice(opts)


def fmt_card(l4: str, rng: random.Random) -> str:
    return rng.choice([f"XX{l4}", f"ending {l4}", f"**** {l4}", f"xx{l4}", f"XXXX-{l4}"])


def rand_ref(inst: str | None, rng: random.Random) -> str:
    if inst in ("NEFT", "RTGS") and rng.random() < 0.6:
        return rng.choice("NHSKBP") + "".join(rng.choice("0123456789") for _ in range(15))
    if rng.random() < 0.15:
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(10))
    return "".join(rng.choice("0123456789") for _ in range(12))


def rand_counterparty(pool: str | None, rng: random.Random) -> str | None:
    if pool is None:
        return None
    if pool == "upi_party":
        r = rng.random()
        if r < 0.4:
            return rng.choice(MERCHANTS)
        if r < 0.7:
            return rng.choice(PERSONS)
        name = rng.choice(PERSONS).lower().split()
        return f"{name[0]}.{name[1][:1]}{rng.randint(1, 99)}{rng.choice(VPA_HANDLES)}"
    if pool == "transfer_party":
        return rng.choice(EMPLOYERS) if rng.random() < 0.5 else rng.choice(PERSONS)
    return rng.choice({"merchant": MERCHANTS, "person": PERSONS, "subscription": SUBSCRIPTIONS,
                       "biller": BILLERS}[pool])


# ---------------------------------------------------------------------------
# Perturbations (edge cases). Each returns (text, tag) and never changes gold values.
# ---------------------------------------------------------------------------
DEV_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def perturb(text: str, lang: str, rng: random.Random, p: float = 0.35) -> tuple[str, list[str]]:
    tags = []
    if rng.random() > p:
        return text, tags
    ops = ["spacing", "header", "promo_tail", "footer"]
    if lang in ("en", "hinglish"):
        ops += ["upper", "lower"]
    if lang == "hi":
        ops += ["dev_digits"]
    for op in rng.sample(ops, k=rng.choice([1, 1, 2])):
        if op == "upper":
            text = text.upper()
        elif op == "lower":
            text = text.lower()
        elif op == "spacing":
            # remove a space between a number and a letter, or double a space
            if rng.random() < 0.5:
                text = re.sub(r"(?<=\d) (?=[^\W\d])", "", text, count=1)
            else:
                text = re.sub(r" ", "  ", text, count=1)
        elif op == "header":
            text = rng.choice(["[Kestrel Bk] ", "BNYN: ", "Alert: ", "Txn Alert- "]) + text
        elif op == "footer":
            text = text + rng.choice([" -Kestrel Bank", " Not you? Report via your bank app.",
                                      " Download our app for details.", " - Neem Bank"])
        elif op == "promo_tail":
            text = text + rng.choice([" Get 5% cashback up to Rs.100 on your next txn!",
                                      " Earn 2X reward points on weekend spends.",
                                      " Win vouchers worth Rs 500. T&C apply."])
        elif op == "dev_digits":
            text = text.translate(DEV_DIGITS)
        tags.append(op)
    return text, tags


# ---------------------------------------------------------------------------
# Rendering one example
# ---------------------------------------------------------------------------
@dataclass
class Example:
    id: str
    family: str
    lang: str
    template_id: str
    sms: str
    gold: dict
    tags: list = field(default_factory=list)
    split: str = ""

    def to_json(self) -> dict:
        return {"id": self.id, "family": self.family, "lang": self.lang,
                "template_id": self.template_id, "sms": self.sms,
                "gold": self.gold, "tags": self.tags, "split": self.split}


_SLOT = re.compile(r"\{(@?)(\w+)(?:=(\w+))?\}")


def render(family: str, lang: str, t_idx: int, rng: random.Random, ex_id: str,
           noise_p: float = 0.35, allow_drop: bool = True) -> Example:
    spec = T[family]
    meta = spec["meta"]
    tmpl = spec[lang][t_idx]
    status = meta.get("status")
    if isinstance(tmpl, tuple):
        tmpl, status = tmpl
    segments = tmpl.split("‖")
    keep = [True] * len(segments)
    tags = []
    if allow_drop and meta["is_txn"] and len(segments) > 1:
        # optional-segment dropping (30%) and truncation (10%)
        for k in range(1, len(segments)):
            if rng.random() < 0.15:
                keep[k] = False
        if any(not x for x in keep[1:]):
            tags.append("missing_fields")
        if rng.random() < 0.10:
            cut = rng.randint(1, len(segments) - 1)
            for k in range(cut, len(segments)):
                keep[k] = False
            tags.append("truncated")

    inst = rng.choice(meta["inst"]) if meta.get("inst") else None
    foreign = meta.get("foreign", 0) and rng.random() < meta["foreign"]
    amount = rand_amount(rng, 1, 3000) if foreign else rand_amount(rng)
    code, sym = rng.choice(FOREIGN) if foreign else ("INR", "₹")
    d = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 730))
    acct = f"{rng.randint(0, 9999):04d}"
    card = f"{rng.randint(0, 9999):04d}"
    values = {
        "amt": fmt_foreign(amount, code, sym, rng) if foreign else fmt_inr(amount, lang, rng),
        "amt2": fmt_inr(round(amount * 0.05, 2) if amount > 100 else amount, lang, rng),
        "acct": fmt_acct(acct, lang, rng),
        "card": fmt_card(card, rng),
        "date": fmt_date(d, rng),
        "time": f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        "cp": rand_counterparty(meta.get("cp"), rng) or "",
        "ref": rand_ref(inst, rng),
        "bal": fmt_inr(rand_amount(rng, 10, 900000), lang, rng),
        "limit": fmt_inr(rand_amount(rng, 1000, 500000), lang, rng),
        "inst": {"AUTOPAY": "AutoPay"}.get(inst, inst) if inst else "",
        "bank": rng.choice(FICTIONAL_BANKS),
        "otp": f"{rng.randint(0, 999999):06d}",
    }
    gold_values = {
        "amt": ("amount", amount), "acct": ("account_last4", acct), "card": ("account_last4", card),
        "date": ("date", d.isoformat()), "cp": ("counterparty", values["cp"] or None),
        "ref": ("reference", values["ref"]), "bal": ("balance", None),
        "inst": ("instrument", inst),
    }
    # balance gold value is the numeric value we rendered; re-derive from text below
    bal_value = None

    revealed = {}
    parts = []
    for seg, k in zip(segments, keep):
        if not k:
            continue

        def sub(m):
            hidden, name, fixed = m.group(1), m.group(2), m.group(3)
            if name in gold_values:
                fname, val = gold_values[name]
                if fixed:
                    val = fixed
                revealed[fname] = val
                if name == "card":
                    revealed["instrument"] = "CARD"
                if name == "amt":
                    revealed["currency"] = code
            if name == "bal":
                revealed["balance"] = "__BAL__"
            return "" if hidden else values[name]
        parts.append(_SLOT.sub(sub, seg))
    text = "".join(parts).strip()
    # numeric balance value from rendered string
    if "balance" in revealed:
        bal_value = parse_rendered_number(values["bal"])
        revealed["balance"] = bal_value
    if text and text[-1] not in ".।!":
        text += "." if lang != "hi" else "।"

    gold = {f: None for f in FIELDS}
    gold["is_transaction"] = bool(meta["is_txn"])
    if meta["is_txn"]:
        gold["type"] = meta["type"]
        gold["status"] = status
        for k, v in revealed.items():
            gold[k] = v
        if foreign:
            tags.append("foreign_currency")
        if "{limit}" in "".join(s for s, k in zip(segments, keep) if k):
            tags.append("limit_distractor")
    else:
        tags.append("hard_negative")
    text, ptags = perturb(text, lang, rng, p=noise_p)
    tags += ptags
    return Example(id=ex_id, family=family, lang=lang, template_id=f"{family}/{lang}/{t_idx}",
                   sms=text, gold=gold, tags=tags)


_NUM = re.compile(r"[\d,]+(?:\.\d+)?")


def parse_rendered_number(s: str) -> float:
    m = _NUM.search(s)
    return float(m.group(0).replace(",", "")) if m else None


# ---------------------------------------------------------------------------
# Dataset builder with template-level held-out split
# ---------------------------------------------------------------------------
def build_dataset(n_train=2400, n_dev=200, n_test_id=400, n_test_ood=400, seed=7,
                  neg_share=0.3, noise_p=0.35, allow_drop=True,
                  exclude_sms: set | None = None) -> dict[str, list[Example]]:
    """noise_p / allow_drop / neg_share = 0 give the 'clean data' ablation."""
    rng = random.Random(seed)
    seen_sms = set(exclude_sms or ())
    out = {"train": [], "dev": [], "test_id": [], "test_ood": []}

    def pick(held_out: bool):
        fam = rng.choice(NEG_FAMILIES) if rng.random() < neg_share else rng.choice(TXN_FAMILIES)
        lang = rng.choice(LANGS)
        n_t = len(T[fam][lang])
        t_idx = n_t - 1 if held_out else rng.randrange(0, n_t - 1)
        return fam, lang, t_idx

    for split, n in [("test_ood", n_test_ood), ("test_id", n_test_id), ("dev", n_dev), ("train", n_train)]:
        i = 0
        tries = 0
        while len(out[split]) < n:
            tries += 1
            fam, lang, t_idx = pick(held_out=(split == "test_ood"))
            ex = render(fam, lang, t_idx, rng, f"{split}-{i:05d}", noise_p=noise_p,
                        allow_drop=allow_drop)
            if ex.sms in seen_sms:      # exact-duplicate guard across ALL splits (no leakage)
                continue
            seen_sms.add(ex.sms)
            ex.split = split
            out[split].append(ex)
            i += 1
    return out


def gold_to_str(gold: dict) -> str:
    return json.dumps({k: gold[k] for k in FIELDS}, ensure_ascii=False)


SYSTEM_PROMPT = (
    "You extract structured data from Indian bank/UPI SMS (English, Hindi or Hinglish). "
    "Reply with ONLY one JSON object with exactly these keys: "
    "is_transaction (true only if money actually moved or a transaction was attempted/failed/reversed; "
    "false for OTPs, offers, reminders, future debits, balance info), "
    "type (\"debit\"|\"credit\"|null), amount (number), currency (ISO code, e.g. \"INR\"), "
    "account_last4 (last 4 digits as string), "
    "instrument (\"UPI\"|\"CARD\"|\"NEFT\"|\"IMPS\"|\"RTGS\"|\"ATM\"|\"NACH\"|\"AUTOPAY\"|null), "
    "counterparty (merchant/person/VPA as written), date (YYYY-MM-DD; SMS dates are day-first), "
    "reference (string), balance (number; available balance only, never credit limit), "
    "status (\"success\"|\"failed\"|\"reversed\"|null). "
    "Use null for anything not stated in the SMS. Never guess. Use ASCII digits. "
    "If is_transaction is false, set every other key to null."
)


def build_messages(sms: str, few_shot: list[tuple[str, str]] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s, g in (few_shot or []):
        msgs.append({"role": "user", "content": f"SMS: {s}"})
        msgs.append({"role": "assistant", "content": g})
    msgs.append({"role": "user", "content": f"SMS: {sms}"})
    return msgs


if __name__ == "__main__":
    ds = build_dataset()
    for k, v in ds.items():
        print(k, len(v))
    for ex in ds["train"][:6]:
        print(ex.lang, ex.family, ex.tags, "\n ", ex.sms, "\n ", gold_to_str(ex.gold))
