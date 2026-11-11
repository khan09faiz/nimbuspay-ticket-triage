"""Part A: audit train.jsonl against SCHEMA.md, write the cleaned set and cleaning_log.csv.

    python audit.py            # reads data/train.jsonl (+ dev/test inputs, for leak check only)

Outputs: data/train_clean.jsonl, cleaning_log.csv, and a summary on stdout.
Every check is a few lines; the business rules live in rules.py.
"""
import collections
import csv
import json
import re

import rules


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm_text(user_text):
    """Ticket text without header, case, spacing and the trailing 'Thx' some copies carry."""
    body = user_text.split("\n\n", 1)[-1].lower()
    body = re.sub(r"\s+", " ", body).strip()
    return re.sub(r"\s*thx$", "", body)


def main():
    train = load("data/train.jsonl")
    heldout = {norm_text(r["messages"][1]["content"])
               for f in ("data/dev.jsonl", "data/test_inputs.jsonl") for r in load(f)}
    log, clean, seen = [], [], {}
    stats = collections.Counter()

    for r in train:
        user, raw = r["messages"][1]["content"], r["messages"][2]["content"]
        _, body = rules.split_ticket(user)

        def drop(reason, key):
            log.append((r["id"], "drop", reason)); stats[key] += 1

        # 1. junk tickets ("test", "asdf asdf", empty) carrying a confident label
        if len(body.strip()) < 15:
            drop(f"junk ticket text {body.strip()!r} with a made-up label", "drop: junk text"); continue
        # 2. label is not valid JSON (cut off mid-string)
        try:
            gold = json.loads(raw)
        except json.JSONDecodeError:
            drop("label is truncated / invalid JSON", "drop: truncated label"); continue
        # 3. same ticket appears in dev/test -> leakage
        if norm_text(user) in heldout:
            drop("ticket text also appears in dev.jsonl (leak)", "drop: dev leak"); continue

        fixes = []
        # 4. legacy (v1) schema: key "prio" with low/medium/high, dates as DD/MM/YYYY
        if "prio" in gold:
            gold = {("priority" if k == "prio" else k): v for k, v in gold.items()}
            gold["priority"] = {"low": "P3", "medium": "P2", "high": "P1"}[gold["priority"]]
            fixes.append("legacy v1 label: prio low/medium/high -> priority P3/P2/P1"); stats["fix: legacy prio key"] += 1
        d = gold["txn_date"]
        if d and re.fullmatch(r"\d{2}/\d{2}/\d{4}", d):
            gold["txn_date"] = f"{d[6:]}-{d[3:5]}-{d[:2]}"
            fixes.append("txn_date DD/MM/YYYY -> YYYY-MM-DD"); stats["fix: legacy date format"] += 1
        if list(gold) != rules.KEYS:
            drop(f"unexpected keys {list(gold)}", "drop: bad keys"); continue

        # 5-7. compare every rule-derivable field with the spec (category is taken as given)
        want = rules.derive(user, gold["category"])
        if gold["txn_id"] != want["txn_id"] and gold["txn_id"] and re.sub(r"[^A-Z0-9]", "", gold["txn_id"].upper()) == want["txn_id"]:
            fixes.append(f"txn_id {gold['txn_id']!r} not normalised -> {want['txn_id']}"); stats["fix: txn_id not normalised"] += 1
            gold["txn_id"] = want["txn_id"]
        a, w = gold["amount_paise"], want["amount_paise"]
        if a != w and a is not None and w is not None and w // 100 == a:
            fixes.append(f"amount_paise {a} is in rupees -> {w} paise"); stats["fix: amount in rupees"] += 1
            gold["amount_paise"] = w
        if gold["priority"] != want["priority"]:
            fixes.append(f"priority {gold['priority']} contradicts spec -> {want['priority']}"); stats["fix: wrong priority"] += 1
            gold["priority"] = want["priority"]
        if gold["needs_human"] != want["needs_human"]:
            fixes.append(f"needs_human {gold['needs_human']} contradicts spec -> {want['needs_human']}"); stats["fix: wrong needs_human"] += 1
            gold["needs_human"] = want["needs_human"]
        # anything the rules still disagree with is something I could not explain -> drop, do not guess
        left = [k for k in rules.KEYS if gold[k] != want[k]]
        if left:
            drop(f"label disagrees with spec on {left} and could not be repaired safely", "drop: unexplained"); continue

        # 8. duplicate tickets (exact, or differing only in spacing / trailing 'Thx')
        key = norm_text(user)
        if key in seen:
            drop(f"duplicate of {seen[key]}", "drop: duplicate"); continue
        seen[key] = r["id"]

        if fixes:
            log.append((r["id"], "fix", "; ".join(fixes)))
        msgs = r["messages"][:2] + [{"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}]
        clean.append({"id": r["id"], "messages": msgs})

    with open("data/train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("cleaning_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "action", "reason"]); w.writerows(log)

    print(f"train rows: {len(train)}  ->  kept: {len(clean)}")
    print(f"rows dropped: {sum(a == 'drop' for _, a, _ in log)}   rows fixed: {sum(a == 'fix' for _, a, _ in log)}")
    for k, v in sorted(stats.items()):
        print(f"  {k:32} {v}")
    cats = collections.Counter(json.loads(r["messages"][2]["content"])["category"] for r in clean)
    print("category balance after cleaning:", dict(cats.most_common()))


if __name__ == "__main__":
    main()
