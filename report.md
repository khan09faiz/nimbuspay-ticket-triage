# Report — NimbusPay ticket triage

Base model `Qwen/Qwen2.5-1.5B-Instruct`, LoRA (r=16, 18.5M trainable parameters), 16-bit, free Colab T4.
Final dev score: **95.0% exact match, 99.0% mean field accuracy** (base model, fixed prompt: 0%).

## 1. Data audit (Part A)

Method: `rules.py` is SCHEMA.md written as code. Given a ticket and its category it derives every other field. `audit.py` runs structural checks, then compares each label with what the spec says. I fixed a row when the right value was unambiguous and dropped it otherwise. All rule code was developed on `train.jsonl` only and is never applied to model outputs.

| # | Problem | How I found it | Rows | Action |
|---|---|---|---|---|
| 1 | Label cut off mid-string (invalid JSON) | `json.loads` on every label | 25 | drop |
| 2 | Junk tickets (`test`, `asdf asdf`, empty) with confident labels | ticket body shorter than 15 characters | 10 | drop |
| 3 | Ticket also present in `dev.jsonl` (leak) | normalised text match against dev and test inputs (0 matches with test) | 45 | drop |
| 4 | Duplicate tickets (exact copies, or differing only in spacing or a trailing "Thx") | normalised text match inside train | 91 | drop later copy |
| 5 | Old label schema: key `prio` with `low/medium/high` | key-set count per label | 70 | fix → `priority` P3/P2/P1 |
| 6 | Old date format `DD/MM/YYYY` (all inside the 70 rows above) | regex on `txn_date` | 30 | fix → `YYYY-MM-DD` |
| 7 | `txn_id` not normalised (`np 81633 47892`, `NP-…`) | regex `NP\d{10}` | 20 | fix |
| 8 | `amount_paise` stored in rupees (also truncates paise: 594.50 → 594) | label = rule amount ÷ 100 | 25 | fix |
| 9 | `priority` contradicts the spec (random flips in both directions) | rule-derived priority ≠ label | 60 | fix |

Result: 1,445 → **1,274 rows** (171 dropped, 175 fixed), all in `cleaning_log.csv`.

Checks that came back clean: `needs_human`, `language`, `channel` and `txn_date` values agree with the rule code on every remaining row. For priority, the label's `needs_human` always agreed with the *rule* priority, which is why I trust the rule over the label there. The mapping low/medium/high → P3/P2/P1 also agreed with the rules on all 70 legacy rows.

Category cannot be derived by rule, so I ran a 10-fold cross-validated TF-IDF + logistic regression and read all of the roughly 60 out-of-fold disagreements. They are classifier mistakes (mostly fraud worded like a refund), not label errors, so I left category untouched. I cannot rule out a few wrong categories that a bag-of-words model also gets "wrong" in the same way.

Not a label problem but important: 36 clean train rows (3 dev, 11 test) are emails with long quoted support threads, up to 2,214 tokens, and the label comes after them.

## 2. Baselines and pipeline (Parts B, C)

| Baseline (dev, 200) | JSON valid | Exact | Field acc |
|---|---|---|---|
| Base model, fixed prompt | 0% | 0% | 0% |
| Base model, SCHEMA.md in system prompt | 62.5% | 0.5% | 37.8% |

The fixed-prompt model wraps its answer in a ```` ```json ```` fence and invents its own keys, so the strict scorer gives 0.

Pipeline: the model's own chat template with exactly the system + user message; loss only on the assistant tokens (JSON + `<|im_end|>`); max length 2,304 because the longest sequence is 2,214 tokens and a 512 or 1,024 limit would cut off the label on every threaded email; plain PyTorch loop, AdamW, lr 2e-4 cosine with 5% warm-up, 3 epochs, batches of up to 12 rows capped at 2,304 padded tokens. Two memory fixes were needed on the T4: gradient checkpointing, and computing logits only at label positions (full logits over a 151k vocabulary for 2k prompt tokens ran out of memory twice before I did this).

## 3. Controlled experiments (Part D)

Each run changes one thing relative to A. Same seed, same loop. Dev = 200 tickets.

| Run | What changed | Exact | Field acc | Peak GPU | Train min |
|---|---|---|---|---|---|
| A | reference: cleaned data, 16-bit LoRA, r=16, lr 2e-4, 3 epochs | 86.0% | 97.3% | 4.79 GB | 12.6 |
| B | raw `train.jsonl` (1,445 rows) | 82.5% | 97.2% | 4.81 GB | 14.3 |
| C | base model in 4-bit NF4 (QLoRA) | 87.0% | 97.7% | 4.03 GB | 15.9 |
| D | rank 4 (4.6M parameters) | 81.5% | 96.5% | 4.63 GB | 12.7 |
| E | + 500 rule-targeted synthetic rows (own idea) | **95.0%** | **99.0%** | 4.80 GB | 17.3 |

- **Cleaning (B vs A): +3.5 points exact.** Field accuracy barely moves because most raw labels are right; the damage is concentrated in priority and amount, which decide exact match. The raw run still produced 100% valid JSON despite 25 truncated labels in training.
- **4-bit vs 16-bit (C vs A): no real difference.** +1 point is 2 tickets out of 200, inside noise for a single seed. 4-bit saved only 0.8 GB (the model is small and activations dominate) and trained 26% slower, so I kept 16-bit for the final model to make reproduction simpler.
- **Rank (D vs A): −4.5 points at r=4.** The task needs computation (date arithmetic, unit conversion, threshold rules), not just an output format, and r=4 has a higher final loss (0.0040 vs 0.0015).
- **Augmentation (E vs A): +9 points.** Each synthetic row is a cleaned ticket plus one inserted sentence whose effect on the label is known by construction: a relative date, an amount in a new format (`k`, `lakh`, `Rs.123/-`), a transaction id with unusual separators, a payment channel, or a distractor (balance, limit, fee, order number). For `kyc`/`other`/`account_access` tickets the inserted detail must leave the transaction fields null. Priority and `needs_human` are recomputed from the rules.

Caveats: one seed per run, 200 dev tickets (1 ticket = 0.5 points), and the final model was selected on dev, so 95% is slightly optimistic.

## 4. Error analysis (Part E) — final model on dev, 10 of 200 wrong

Errors per field: priority 8, needs_human 4 (all follow from a wrong priority), category 3, amount 1. Dates, ids, channel and language: 100%. Weakest categories: **fraud 86.7%** exact (only 75 training rows) and **refund 89.1%**. Hinglish 98.0%, English 94.0%.

| id | Error | Why |
|---|---|---|
| dv-00011 | ₹84,137 refund → P2, should be P1 | amount extracted correctly, compared wrongly with the ₹50,000 threshold |
| dv-00037 | "284669 rupees" → P2, should be P1 | same; six digits with no separators |
| dv-00056 | Rs 44,206.75 → P1, should be P2 | over-triggers P1 just below ₹50,000 |
| dv-00148 | INR 42,062 → P1, should be P2 | same |
| dv-00044 | Rs. 8377 refund → P3, should be P2 | misses the ₹5,000 threshold |
| dv-00127 | ₹4,836 → P2, should be P3 | over-triggers just below ₹5,000 |
| dv-00017 | Rs 1,80,827.25 → 18082525 | digit copy error in a long Indian-format number (7 became 5) |
| dv-00093 | "A random charge got added to my card" → refund, gold fraud | non-recognition is implied, not stated; fraud is under-represented |
| dv-00157 | "Kisi ne OTP manag ke mere piase nikal liye" → account_access, gold fraud | keyed on "OTP"; typos hide "paise nikal liye" |
| dv-00177 | "red flag on my top up" → fraud, gold payment_failed | "red flag" read as suspicious activity |

Main pattern: **6 of 10 failures are numeric comparisons against the ₹5,000 / ₹50,000 thresholds**, mostly within ~20% of the threshold. The model reads the amount correctly and then misjudges its size in a single forward pass.

Test set (no labels): 399 of 400 outputs are valid JSON with the 8 keys. `ts-00079` stopped after two keys (`{"category": "fraud", "priority": "P1"}`) and will score zero; I left it as generated. Reading test outputs by hand, the same weaknesses show up more often than on dev: priority near thresholds, relative dates that need arithmetic across a month boundary ("5 din pehle", "last month on the 29th"), and one case where a convenience fee was taken as the amount. I expect the hidden-test score to be clearly below the dev score.

Not captured: the notebook's last two cells (20-row reproducibility self-check and the output bundle) ran, but the browser session dropped before their output was saved, so the self-check result is not shown in the notebook.

## 5. With one more week

1. More targeted augmentation for the two measured weaknesses: amounts sampled densely around ₹5,000 and ₹50,000 in every surface format, and relative dates crossing month/year boundaries.
2. More fraud examples built from train's fraud tickets with implicit wording, since fraud overrides every other category and has the fewest rows.
3. Three seeds per configuration and a held-out slice of train for model selection, so dev stays a clean estimate.
4. Try a 3–4B model (same pipeline) for the arithmetic-heavy fields, and check 4-bit vs 16-bit on that model, where memory actually matters.
5. Find out why `ts-00079` stopped early.

## 6. AI tools

I used Claude (Anthropic, via Claude Code) throughout: to write the audit, rule and augmentation code, to assemble and debug the notebook (including the two out-of-memory fixes), to drive the Colab run, and to draft this report. All numbers come from the notebook outputs and `results.json`. I also read the unlabeled test inputs to check length and wording shift; no dev or test row was used for training.
