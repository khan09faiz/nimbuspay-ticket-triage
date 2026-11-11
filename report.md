# Report: NimbusPay ticket triage

Base model `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 (18.5M trainable parameters), 16-bit, free Colab T4.
Final dev score: **95.0% exact match, 99.0% mean field accuracy**. The base model with the fixed prompt scores 0%.

## 1. Data audit (Part A)

I wrote SCHEMA.md as code in `rules.py`. Given a ticket and its category, it works out every other field. `audit.py` runs structural checks first and then compares each label with what the spec says. I fixed a row when the right value was obvious and dropped it when it was not. I built the rule code using `train.jsonl` only, and it is never run on model outputs.

| # | Problem | How I found it | Rows | Action |
|---|---|---|---|---|
| 1 | Label cut off in the middle (invalid JSON) | `json.loads` on every label | 25 | drop |
| 2 | Junk tickets (`test`, `asdf asdf`, empty) that still have a full label | ticket body under 15 characters | 10 | drop |
| 3 | Same ticket also in `dev.jsonl` (leak) | matched normalised text against dev and test inputs. No match with test | 45 | drop |
| 4 | Duplicate tickets, exact or differing only in spacing or a trailing "Thx" | normalised text match inside train | 91 | drop the later copy |
| 5 | Old label format: key `prio` with `low/medium/high` | counted the key sets of all labels | 70 | fix to `priority` P3/P2/P1 |
| 6 | Old date format `DD/MM/YYYY`, all inside the 70 rows above | regex on `txn_date` | 30 | fix to `YYYY-MM-DD` |
| 7 | `txn_id` not normalised (`np 81633 47892`, `NP-...`) | regex `NP\d{10}` | 20 | fix |
| 8 | `amount_paise` given in rupees. This also loses the paise: 594.50 became 594 | label equals rule amount divided by 100 | 25 | fix |
| 9 | `priority` does not follow the spec, wrong in both directions | priority from the rules differs from the label | 60 | fix |

Result: 1,445 rows became **1,274** (171 dropped, 175 fixed). Every change is in `cleaning_log.csv`.

Some checks found nothing. `needs_human`, `language`, `channel` and `txn_date` agree with the rule code on every remaining row. In the 60 rows with a wrong priority, the `needs_human` label still matched the priority from the rules, so I trusted the rule and not the label. The low/medium/high mapping also matched the rules on all 70 old-format rows.

Category cannot be worked out by a rule. To check it I ran a 10-fold cross-validated TF-IDF and logistic regression model and read the roughly 60 rows where it disagreed with the label. They were mistakes of the classifier, mostly fraud tickets worded like refunds, so I left category alone. A few wrong categories could still be hiding if a bag-of-words model makes the same mistake.

One more thing I noticed, which is not a label problem: 36 clean train rows (3 in dev, 11 in test) are emails with long quoted support threads. They go up to 2,214 tokens and the label comes after all of it.

## 2. Baselines and pipeline (Parts B, C)

| Baseline (dev, 200 tickets) | JSON valid | Exact | Field acc |
|---|---|---|---|
| Base model, fixed prompt | 0% | 0% | 0% |
| Base model, SCHEMA.md in system prompt | 62.5% | 0.5% | 37.8% |

With the fixed prompt the model wraps its answer in a ```` ```json ```` block and makes up its own keys, so the strict scorer gives it 0.

Pipeline:
- The model's own chat template with exactly the system and user message.
- Loss only on the assistant tokens (the JSON and `<|im_end|>`).
- Max length 2,304. The longest sequence is 2,214 tokens, and a limit of 512 or 1,024 would cut off the label on every threaded email.
- Plain PyTorch loop, AdamW, lr 2e-4 with cosine decay and 5% warm-up, 3 epochs, batches of up to 12 rows capped at 2,304 padded tokens.
- Two memory fixes for the T4: gradient checkpointing, and computing logits only at label positions. Before that, full logits over a 151k vocabulary for 2k prompt tokens ran out of memory twice.

## 3. Controlled experiments (Part D)

Each run changes one thing compared with A. Same seed and same loop. Dev has 200 tickets.

| Run | What changed | Exact | Field acc | Peak GPU | Train min |
|---|---|---|---|---|---|
| A | reference: cleaned data, 16-bit LoRA, r=16, lr 2e-4, 3 epochs | 86.0% | 97.3% | 4.79 GB | 12.6 |
| B | raw `train.jsonl` (1,445 rows) | 82.5% | 97.2% | 4.81 GB | 14.3 |
| C | base model in 4-bit NF4 (QLoRA) | 87.0% | 97.7% | 4.03 GB | 15.9 |
| D | rank 4 (4.6M parameters) | 81.5% | 96.5% | 4.63 GB | 12.7 |
| E | added 500 synthetic rows (my own idea) | **95.0%** | **99.0%** | 4.80 GB | 17.3 |

- **Cleaning (B vs A): +3.5 points exact.** Field accuracy hardly moves because most raw labels are correct. The bad labels are mostly in priority and amount, and one wrong field is enough to lose the exact match. The raw run still gave 100% valid JSON even with 25 truncated labels in training.
- **4-bit vs 16-bit (C vs A): no real difference.** One point is 2 tickets out of 200, which is noise for a single seed. 4-bit saved only 0.8 GB because the model is small and activations take most of the memory, and it trained 26% slower. I kept 16-bit for the final model so it is simpler to reproduce.
- **Rank (D vs A): 4.5 points lower at r=4.** The model has to learn date arithmetic, unit conversion and threshold rules, and r=4 does not have enough capacity for that. Its final loss is 0.0040 against 0.0015 for r=16.
- **Synthetic rows (E vs A): +9 points.** Each synthetic row is a cleaned ticket with one extra sentence, and I know how that sentence changes the label. The sentence is a relative date, an amount in a new format (`k`, `lakh`, `Rs.123/-`), a transaction id with odd separators, a payment channel, or a number that must be ignored (balance, limit, fee, order number). For `kyc`, `other` and `account_access` tickets the transaction fields stay null whatever is inserted. Priority and `needs_human` are recomputed from the rules.

Limits of these numbers: one seed per run, 200 dev tickets so one ticket is 0.5 points, and I picked the final model on dev, so 95% is a little optimistic.

## 4. Error analysis (Part E)

The final model gets 10 of 200 dev tickets wrong.

Errors per field: priority 8, needs_human 4 (each one follows from a wrong priority), category 3, amount 1. Dates, ids, channel and language are all 100%. The weakest categories are **fraud at 86.7%** exact (only 75 training rows) and **refund at 89.1%**. Hinglish tickets score 98.0% and English 94.0%.

| id | Error | Why I think it happened |
|---|---|---|
| dv-00011 | ₹84,137 refund given P2, should be P1 | amount is read correctly but compared wrongly with ₹50,000 |
| dv-00037 | "284669 rupees" given P2, should be P1 | same problem, six digits with no separators |
| dv-00056 | Rs 44,206.75 given P1, should be P2 | jumps to P1 just under ₹50,000 |
| dv-00148 | INR 42,062 given P1, should be P2 | same |
| dv-00044 | Rs. 8377 refund given P3, should be P2 | misses the ₹5,000 threshold |
| dv-00127 | ₹4,836 given P2, should be P3 | jumps to P2 just under ₹5,000 |
| dv-00017 | Rs 1,80,827.25 became 18082525 | copied one digit wrong in a long Indian-format number (7 became 5) |
| dv-00093 | "A random charge got added to my card" given refund, gold is fraud | the customer does not say outright that they did not make it |
| dv-00157 | "Kisi ne OTP manag ke mere piase nikal liye" given account_access, gold is fraud | it latched onto "OTP", and the typos hide "paise nikal liye" |
| dv-00177 | "red flag on my top up" given fraud, gold is payment_failed | "red flag" was read as suspicious activity |

The main pattern: **6 of the 10 failures are comparisons against the ₹5,000 or ₹50,000 threshold**, mostly for amounts within about 20% of it. The model reads the amount correctly and then gets the comparison wrong.

Test set (no labels): 399 of 400 outputs are valid JSON with the 8 keys. `ts-00079` stopped after two keys (`{"category": "fraud", "priority": "P1"}`) and will score zero. I left it as the model produced it. I read through the test outputs and saw the same weak spots more often than on dev: priority near the thresholds, relative dates that cross a month boundary ("5 din pehle", "last month on the 29th"), and one ticket where a convenience fee was taken as the amount. So I expect the hidden test score to be lower than the dev score.

One gap in the notebook: the last two cells (a 20-row reproducibility check and the zip of outputs) did run, but my browser session dropped before their output was saved, so the result of the check is not shown.

## 5. With one more week

1. More synthetic rows for the two weak spots I measured: amounts placed close to ₹5,000 and ₹50,000 in every format, and relative dates that cross a month or year boundary. I started on the first one but ran out of free Colab GPU time before the run finished.
2. More fraud examples with indirect wording, built from the fraud tickets in train. Fraud overrides every other category and has the fewest rows.
3. Three seeds per setting, and a held-out part of train for picking the model, so dev stays a clean estimate.
4. Try a 3B to 4B model with the same pipeline for the fields that need arithmetic, and compare 4-bit with 16-bit there, where memory actually matters.
5. Find out why `ts-00079` stopped early.

## 6. AI tools

I used Claude (Anthropic) as a coding assistant throughout this assignment. It helped write the audit, rule and augmentation code, put together and debug the notebook (including the two out-of-memory fixes), run it on Colab, and draft this report. All numbers come from the notebook outputs and `results.json`. I also read the unlabeled test inputs to check ticket length and wording. No dev or test row was used for training.
