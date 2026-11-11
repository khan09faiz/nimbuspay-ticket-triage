"""SCHEMA.md written as code.

Given a ticket and its category, derive every other field of the triage record.
Used ONLY to audit / repair train.jsonl and to label augmented training rows.
It is never used at inference time (the assignment forbids post-processing).
"""
import re
from datetime import date, timedelta

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
TXN_CATEGORIES = {"payment_failed", "refund", "fraud", "offers"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
            r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")

LEGAL_RE = re.compile(r"\b(rbi|ombudsman|consumer court|court|lawyer|legal notice|legal action|legal)\b", re.I)
REPEAT_RE = re.compile(
    r"third time|second complaint|again and again|already written|contacted support about this before|"
    r"baar baar|teesri baar|dusri baar|pehle bhi complaint|followed up|\b\d+(st|nd|rd|th) time\b", re.I)
# sentences whose rupee amounts are distractors (balance, limits, fees)
DISTRACTOR_RE = re.compile(r"balance|limit|\bfee\b|\bfees\b|pade hain", re.I)
HINGLISH = set("""hai hain hoon hua hui tha thi karo karna karun kiya kiye raha rahi rahe nahi nahin
mera meri mere mujhe maine mein ka ki aur lekin bhai kya kyun kaise abhi bhi gaya gayi gaye
paise paisa rupaye wapas chahiye baar pehle parso aaj yeh woh wala wali dhanyavaad namaste jaldi
kat liye liya diya dunga jaunga bhool badalne baat rakam poore mamla likha chuka kuch""".split())


def split_ticket(user_text):
    """-> (received_date, customer_message_without_quoted_lines)"""
    m = re.match(r"Received: (\d{4})-(\d{2})-(\d{2})", user_text)
    received = date(int(m[1]), int(m[2]), int(m[3]))
    body = user_text.split("\n\n", 1)[1] if "\n\n" in user_text else ""
    body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">"))
    return received, body


def txn_id(body):
    m = re.search(r"\bN\s?P[\s\-]*((?:\d[\s\-]?){10})(?!\d)", body, re.I)
    return "NP" + re.sub(r"\D", "", m[1]) if m else None


def _strip_noise(body):
    """Remove things that contain digits but are not amounts or dates."""
    b = re.sub(r"\bN\s?P[\s\-]*(?:\d[\s\-]?){10}", " ", body, flags=re.I)       # txn ids
    b = re.sub(r"#\d+|\b\d{9,}\b", " ", b)                                       # ticket / bank refs
    b = re.sub(r"(order number|order no\.?|order id)\s*\S+", " ", b, flags=re.I)
    b = re.sub(r"version (is )?\d+(\.\d+)+|\b\d+\.\d+\.\d+\b", " ", b, flags=re.I)  # app versions
    b = re.sub(r"(iPhone|Redmi Note|Realme|Samsung M|Vivo Y|Pixel|OnePlus)\s?\d+\w*", " ", b)
    return b


def _num(s):
    return float(s.replace(",", ""))


def amount_paise(body):
    b = _strip_noise(body)
    # drop distractor sentences; "Rs." must not count as a sentence end
    b = re.sub(r"\bRs\.", "Rs", b)
    # distractor clauses sometimes follow the real amount without a full stop
    b = re.sub(r"(FYI|My account balance|Account mein|Meri daily|A convenience fee)", r". \1", b)
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", b) if not DISTRACTOR_RE.search(s)]
    b = " ".join(sents)
    b = _DATE_STRIP.sub(" ", b)
    num = r"(\d[\d,]*(?:\.\d+)?)"
    cands = []
    for m in re.finditer(num + r"\s?(k|lakhs?|lacs?)\b", b, re.I):
        cands.append((m.start(), _num(m[1]) * (1000 if m[2].lower() == "k" else 100000)))
    if not cands:
        for m in re.finditer(r"(?:₹|\bRs\b\.?|\bINR\b)\s*" + num, b, re.I):
            cands.append((m.start(), _num(m[1])))
    if not cands:
        for m in re.finditer(num + r"\s?(?:rupees|rupaye|rs\b|/-)", b, re.I):
            cands.append((m.start(), _num(m[1])))
    if not cands:  # bare number with an amount cue
        for m in re.finditer(num + r"(?= (?:is the amount|was the value|in total|kat gaye|tha\b|ka tha|ka transaction))", b, re.I):
            cands.append((m.start(), _num(m[1])))
        for m in re.finditer(r"(?:amount|total|rakam|value)\D{0,12}" + num, b, re.I):
            cands.append((m.start(), _num(m[1])))
    if not cands:
        return None
    return int(round(cands[0][1] * 100))


_DATE_PATTERNS = [
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + MONTH_RE + r"\b(?:\s+\d{4})?",
    r"\b" + MONTH_RE + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?:,?\s+\d{4})?",
]
_DATE_STRIP = re.compile("|".join(_DATE_PATTERNS) + r"|\b\d+\s+(?:days? ago|din pehle)", re.I)


def _most_recent(received, month, day):
    for y in (received.year, received.year - 1):
        try:
            d = date(y, month, day)
        except ValueError:
            continue
        if d <= received:
            return d
    return None


def txn_date(body, received):
    b = _strip_noise(body)
    low = b.lower()
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", b)
    if m:
        y = int(m[3]); y += 2000 if y < 100 else 0
        return date(y, int(m[2]), int(m[1]))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + MONTH_RE + r"\b(?:\s+(\d{4}))?", b, re.I)
    if not m:
        m2 = re.search(r"\b" + MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?", b, re.I)
        if m2:
            m = type("M", (), {"g": (m2[2], m2[1], m2[3])})
            day, mon, yr = m.g
        else:
            day = None
    else:
        day, mon, yr = m[1], m[2], m[3]
    if day:
        mon = MONTHS[mon[:3].lower()]
        return date(int(yr), mon, int(day)) if yr else _most_recent(received, mon, int(day))
    m = re.search(r"\b(\d+)\s+(?:days? ago|din pehle)", low)
    if m:
        return received - timedelta(days=int(m[1]))
    if re.search(r"day before yesterday|\bparso\b", low):
        return received - timedelta(days=2)
    if re.search(r"\byesterday\b|\bkal\b|last night", low):
        return received - timedelta(days=1)
    if re.search(r"\btoday\b|\baaj\b|this morning", low):
        return received
    m = re.search(r"last month on the (\d{1,2})", low)
    if m:
        first = received.replace(day=1) - timedelta(days=1)
        return first.replace(day=int(m[1]))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th) of this month", low)
    if m:
        return received.replace(day=int(m[1]))
    return None


def channel(body):
    low = body.lower()
    hits = []
    for name, pat in (("upi", r"\bupi\b"),
                      ("netbanking", r"net[\s-]?banking|internet banking"),
                      ("wallet", r"\bwallet\b"),
                      ("card", r"\bcard\b|\bvisa\b|mastercard")):
        m = re.search(pat, low)
        if m:
            hits.append((m.start(), name))
    # "scratch card" / "card details" are not a payment mode
    low2 = re.sub(r"\bs[a-z]{5,6} card\b|gift card", " ", low)   # "scratch card" (incl. typos)
    hits = [(p, n) for p, n in hits if n != "card" or re.search(r"\bcard\b|\bvisa\b|mastercard", low2)]
    return sorted(hits)[-1][1] if hits else None   # explicit "Mode:" sentences come after the issue sentence


def language(body):
    words = re.findall(r"[a-z]+", body.lower())
    return "hi-en" if sum(w in HINGLISH for w in words) >= 2 else "en"


def derive(user_text, category):
    received, body = split_ticket(user_text)
    rec = {"category": category, "priority": None, "amount_paise": None, "txn_date": None,
           "txn_id": None, "channel": None, "language": language(body), "needs_human": None}
    if category in TXN_CATEGORIES:
        rec["amount_paise"] = amount_paise(body)
        d = txn_date(body, received)
        rec["txn_date"] = d.isoformat() if d else None
        rec["txn_id"] = txn_id(body)
        rec["channel"] = channel(body)
    return finalize(rec, body)


def finalize(rec, body):
    """Fill priority and needs_human from category, amount and the ticket text."""
    category, amt = rec["category"], rec["amount_paise"] or 0
    if category == "fraud" or amt >= 50_000_00 or LEGAL_RE.search(body):
        rec["priority"] = "P1"
    elif (category in ("payment_failed", "refund") and amt >= 5_000_00) or category == "account_access":
        rec["priority"] = "P2"
    else:
        rec["priority"] = "P3"
    rec["needs_human"] = rec["priority"] == "P1" or bool(REPEAT_RE.search(body))
    return rec
