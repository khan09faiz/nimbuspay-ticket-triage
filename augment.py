"""Experiment 4 (own idea): rule-targeted augmentation.

The cleaned train set states almost every date / amount / id in a handful of surface forms.
SCHEMA.md asks for more than that (relative dates, missing years, k / lakh, ids with spaces and
symbols, distracting amounts and reference numbers, null transaction fields for non-transaction
categories). Each synthetic row = one cleaned train ticket + ONE inserted sentence whose effect
on the label is known by construction. Uses only train.jsonl and SCHEMA.md.
"""
import json
import random
import re
from datetime import date, timedelta

import rules

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _indian(n):
    """123456 -> '1,23,456' (Indian digit grouping)."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    return ",".join(([head] if head else []) + parts + [tail])


def _ord(n):
    return f"{n}{'th' if 10 < n % 100 < 14 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _insert(user, sentence, rng):
    """Insert a sentence at a random sentence boundary of the customer's main paragraph."""
    head, body = user.split("\n\n", 1)
    lines = body.split("\n")
    cand = [i for i, l in enumerate(lines)
            if l.strip() and not l.startswith((">", "Subject:", "Regards")) and re.search(r"[.!?]", l)]
    i = max(cand, key=lambda j: len(lines[j]))
    sents = re.split(r"(?<=[.!?]) +", lines[i])
    k = rng.randint(1, len(sents))
    lines[i] = " ".join(sents[:k] + [sentence] + sents[k:])
    return head + "\n\n" + "\n".join(lines)


def _new_id(rng):
    d = "".join(rng.choice("0123456789") for _ in range(10))
    shown = rng.choice([f"N P {d[:4]} {d[4:7]} {d[7:]}", f"Np-{d[:5]}-{d[5:]}", f"np.{d}", f"NP#{d}",
                        f"nP {d[:2]} {d[2:6]} {d[6:]}", f"NP_{d[:5]}_{d[5:]}", f"np-{d[:3]}-{d[3:6]}-{d[6:]}"])
    return shown, "NP" + d


def op_id(user, gold, hi, rng):
    shown, norm = _new_id(rng)
    m = re.search(r"\bN\s?P[\s\-]*(?:\d[\s\-]?){10}", user, re.I)
    if gold["txn_id"] and m:                       # same ticket, id written in a new way
        return user[:m.start()] + shown + " " + user[m.end():].lstrip(), {"txn_id": norm}
    if gold["txn_id"] is None:
        s = rng.choice(["Iska transaction number {} hai.", "App mein id {} dikha raha hai."] if hi else
                       ["Payment reference: {}.", "The app shows id {} for it.", "NimbusPay id on the SMS is {}."])
        return _insert(user, s.format(shown), rng), {"txn_id": norm}


def op_date(user, gold, hi, rng):
    if gold["txn_date"] is not None:
        return None
    rec, _ = rules.split_ticket(user)
    kind = rng.choice(["night", "week", "ndays", "lastmonth", "thismonth", "noyear", "prevyear", "parso"])
    if kind == "night":
        d, s = rec - timedelta(1), ("Kal raat ko hua yeh." if hi else "This happened last night.")
    elif kind == "week":
        d, s = rec - timedelta(7), ("Ek hafta pehle hua tha." if hi else "It happened exactly a week ago.")
    elif kind == "ndays":
        n = rng.randint(2, 40)
        d, s = rec - timedelta(n), (f"{n} din pehle ki baat hai." if hi else f"That was {n} days back.")
    elif kind == "parso":
        d, s = rec - timedelta(2), ("Parso hua tha yeh." if hi else "It was the day before yesterday.")
    elif kind == "lastmonth":
        last = rec.replace(day=1) - timedelta(1)
        d = last.replace(day=rng.randint(1, last.day))
        s = f"Pichle mahine ki {d.day} tareekh ko hua." if hi else f"It was on the {_ord(d.day)} of last month."
    elif kind == "thismonth":
        d = rec.replace(day=rng.randint(1, rec.day))
        s = f"Isi mahine ki {d.day} tareekh ko hua." if hi else f"It happened on the {_ord(d.day)} of this month."
    else:  # day + month without a year -> most recent such date on or before Received
        back = rng.randint(3, 120) if kind == "noyear" else rng.randint(200, 350)
        d = rec - timedelta(back)
        form = rng.choice([f"{d.day} {MON[d.month - 1]}", f"{MON[d.month - 1]} {d.day}"])
        s = f"{form} ko hua tha." if hi else f"The date was {form}."
    return _insert(user, s, rng), {"txn_date": d.isoformat()}


def op_amount(user, gold, hi, rng):
    if gold["amount_paise"] is not None:
        return None
    kind = rng.choice(["slash", "space", "k", "lakh", "plain", "comma", "paise"])
    if kind == "k":
        x = rng.choice([rng.randint(1, 99), rng.randint(10, 990) / 10]); rupees, shown = x * 1000, f"{x}k"
    elif kind == "lakh":
        x = rng.choice([1, 2, 3, 1.5, 2.5, 0.5, 0.75, 1.2]); rupees, shown = x * 100000, f"{x} lakh"
    else:
        n = rng.choice([rng.randint(20, 999), rng.randint(1000, 9999), rng.randint(10000, 99999), rng.randint(100000, 400000)])
        rupees = n
        shown = {"slash": f"Rs.{n}/-", "space": f"₹ {n}.00", "plain": f"{n} rupees", "comma": f"INR {_indian(n)}"}.get(kind)
        if kind == "paise":
            p = rng.choice([25, 50, 75, 99]); rupees = n + p / 100; shown = f"Rs {_indian(n)}.{p}"
    s = rng.choice(["{} ka payment tha.", "Kul {} ka mamla hai."] if hi else
                   ["The sum in question is {}.", "I am out {} because of this.", "{} is what it was for."])
    return _insert(user, s.format(shown), rng), {"amount_paise": int(round(rupees * 100))}


def op_channel(user, gold, hi, rng):
    if gold["channel"] is not None:
        return None
    ch = rng.choice(["upi", "card", "netbanking", "wallet"])
    s = rng.choice({
        "upi": ["UPI app se bheja tha."] if hi else ["I paid using the UPI app.", "Payment mode was UPI."],
        "card": ["Visa debit card se pay kiya tha."] if hi else ["I used my Mastercard for it.", "It was paid by Visa.", "I paid by credit card."],
        "netbanking": ["Net-banking se kiya tha."] if hi else ["It was done from net-banking.", "I paid through internet banking."],
        "wallet": ["NimbusPay wallet se gaya tha."] if hi else ["It went out of my wallet.", "I paid from the wallet."],
    }[ch])
    return _insert(user, s, rng), {"channel": ch}


def op_distractor(user, gold, hi, rng):
    """Numbers that must be ignored: label does not change."""
    n = rng.randint(50, 90000)
    s = rng.choice(
        [f"Order number {rng.randint(10000, 99999)} hai.", f"Meri daily limit ₹{_indian(n)} hai.",
         f"Wallet mein abhi {n} rupaye bache hain, balance check kar lo.", f"Upar se Rs. {rng.randint(5, 99)} ki fee bhi lagi."] if hi else
        [f"Order number {rng.randint(10000, 99999)} if that helps.", f"My transfer limit is ₹{_indian(n)}.",
         f"My balance shows Rs. {n} at the moment.", f"A convenience fee of Rs. {rng.randint(5, 99)} was also added.",
         f"The bank reference number is {rng.randint(10**11, 10**12 - 1)}.", f"My old ticket was #{rng.randint(10000, 99999)}."])
    return _insert(user, s, rng), {}


OPS = [op_id, op_date, op_amount, op_channel, op_distractor]


def augment(clean_rows, n=500, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        r = rng.choice(clean_rows)
        user = r["messages"][1]["content"]
        gold = json.loads(r["messages"][2]["content"])
        hi = gold["language"] == "hi-en"
        if gold["category"] in rules.TXN_CATEGORIES:
            res = rng.choice(OPS)(user, gold, hi, rng)
        else:
            # non-transaction categories: transaction details may be mentioned, fields stay null
            g2 = dict(gold, amount_paise=None, txn_date=None, txn_id=None, channel=None)
            res = rng.choice([op_id, op_date, op_channel, op_distractor])(user, g2, hi, rng)
            if res:
                res = (res[0], {})
        if not res:
            continue
        new_user, change = res
        gold.update(change)
        _, body = rules.split_ticket(new_user)
        gold = rules.finalize(gold, body)          # amount may change priority / needs_human
        out.append({"id": f"aug-{len(out):04d}-from-{r['id']}",
                    "messages": [r["messages"][0], {"role": "user", "content": new_user},
                                 {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}]})
    return out


if __name__ == "__main__":
    clean = [json.loads(l) for l in open("data/train_clean.jsonl", encoding="utf-8")]
    rows = augment(clean)
    with open("data/train_aug.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "synthetic rows written to data/train_aug.jsonl")
    for r in rows[:6]:
        print("\n" + r["messages"][1]["content"].split("\n> ")[0], "\n ->", r["messages"][2]["content"])
