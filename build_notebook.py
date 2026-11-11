"""Assemble notebook.ipynb from the cells below + rules.py / audit.py / augment.py (embedded via %%writefile)."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
def embed(name): code(f"%%writefile {name}\n" + open(name, encoding="utf-8").read())

md("""
# NimbusPay ticket triage — LoRA fine-tune of Qwen2.5-1.5B-Instruct

Runs top to bottom on a free Colab **T4**. It audits `train.jsonl`, measures two baselines, runs five
controlled fine-tuning experiments, analyses errors on dev and writes `adapter/` + `predictions_test.jsonl`.

**Stack:** plain PyTorch training loop + Hugging Face `transformers` + `peft` (+ `bitsandbytes` for the 4-bit run).
I chose a hand-written loop over Unsloth/TRL because the whole pipeline (chat template, label masking, batching)
is ~40 visible lines that I can explain and change live, and it has no version-sensitive trainer wrappers.

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B params, Apache-2.0, not gated, has a system role in its chat
template, multilingual so Hinglish is not out of distribution). Small enough that 16-bit LoRA *and* 4-bit QLoRA
both fit a T4 and every experiment takes minutes.
""")
code("""
!pip -q install bitsandbytes peft
import os, json, zipfile, shutil, time, gc, random, math, collections
import torch
print(torch.__version__, torch.cuda.get_device_name(0))
""")
md("## 0. Data\nPut `training_assessment.zip` (the candidate pack) next to the notebook, or upload it when asked.")
code("""
if not os.path.exists("data/train.jsonl"):
    if not os.path.exists("training_assessment.zip"):
        from google.colab import files
        files.upload()                      # choose training_assessment.zip
    zipfile.ZipFile("training_assessment.zip").extractall("pack")
    shutil.copytree("pack/candidate_pack/data", "data", dirs_exist_ok=True)
    for f in ("score.py", "SCHEMA.md"):
        shutil.copy(f"pack/candidate_pack/{f}", f)

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

raw_train, dev, test = load("data/train.jsonl"), load("data/dev.jsonl"), load("data/test_inputs.jsonl")
print(len(raw_train), len(dev), len(test))
""")
md("""
## Part A — Audit of `train.jsonl`

`rules.py` is SCHEMA.md written as code: given a ticket and its category it derives every other field.
`audit.py` runs structural checks and then compares each label with what the spec says. Anything that disagrees
is either repaired (when the correct value is unambiguous) or dropped. Every change goes to `cleaning_log.csv`.

The rule code is used **only** on training data. It is never applied to model outputs.
""")
embed("rules.py")
embed("audit.py")
code("!python audit.py")
code("""
# Evidence: one raw example per problem type
import csv, re
log = list(csv.DictReader(open("cleaning_log.csv", encoding="utf-8")))
by_id = {r["id"]: r for r in raw_train}
seen = set()
for row in log:
    kind = re.sub(r"[\\d'\\"].*", "", row["reason"])[:28]
    if kind in seen: continue
    seen.add(kind)
    r = by_id[row["id"]]
    print(f"[{row['action']}] {row['id']}: {row['reason']}")
    print("   ticket:", r["messages"][1]["content"].split("\\n\\n", 1)[-1][:160].replace("\\n", " | "))
    print("   label :", r["messages"][2]["content"][:190], "\\n")
""")
code("""
# How good is the category label? 10-fold cross-validated TF-IDF + logistic regression on the cleaned rows.
# Rows where an out-of-fold classifier confidently disagrees are candidates for label noise. I read them all:
# they are classifier mistakes (mostly fraud vs refund wording), not wrong labels, so category is left untouched.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import numpy as np
clean = load("data/train_clean.jsonl")
X = [r["messages"][1]["content"].split("\\n\\n", 1)[-1].split("\\n>")[0] for r in clean]
y = np.array([json.loads(r["messages"][2]["content"])["category"] for r in clean])
P = cross_val_predict(LogisticRegression(C=20, max_iter=2000), TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(X), y, cv=10, method="predict_proba")
pred = np.array(sorted(set(y)))[P.argmax(1)]
print(f"out-of-fold agreement with category labels: {(pred == y).mean():.1%}; disagreements: {(pred != y).sum()}")
for i in np.argsort(-P.max(1) * (pred != y))[:8]:
    print(f"  {P.max(1)[i]:.2f} label={y[i]:15} clf={pred[i]:15} {X[i][:110]!r}")
""")
md("""
## Part C — Training pipeline

* **Chat template:** the model's own (`tokenizer.apply_chat_template`), with exactly the system + user message from the data.
* **Loss:** only on the assistant tokens (the JSON + `<|im_end|>`). Prompt tokens get label `-100`.
* **Sequence length:** chosen from the token-length evidence printed below.
""")
code("""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import score                                           # the provided scorer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 0
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"                              # only matters for batched generation
EOS = tok.convert_tokens_to_ids("<|im_end|>")

def prompt_text(row, system_extra=""):
    msgs = [dict(m) for m in row["messages"][:2]]
    msgs[0]["content"] += system_extra
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def encode(row):
    p = tok(prompt_text(row), add_special_tokens=False)["input_ids"]
    a = tok(row["messages"][2]["content"], add_special_tokens=False)["input_ids"] + [EOS]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}

print(prompt_text(dev[0]) + dev[0]["messages"][2]["content"] + "<|im_end|>")
""")
code("""
# Sequence-length evidence (prompt + label tokens)
def lens(rows): return np.array([len(encode(r)["input_ids"]) if len(r["messages"]) > 2 else len(tok(prompt_text(r))["input_ids"]) + 75 for r in rows])
for name, rows in (("train(clean)", clean), ("dev", dev), ("test (+75 for label)", test)):
    L = lens(rows)
    print(f"{name:22} median {int(np.median(L)):4d}  p95 {int(np.percentile(L, 95)):4d}  max {L.max():4d}   >512: {(L > 512).sum():3d}   >1024: {(L > 1024).sum():3d}   >2048: {(L > 2048).sum():2d}   >2304: {(L > 2304).sum()}")
MAX_LEN = 2304
# Emails carry long quoted threads AFTER the customer's text. The label is the last ~70 tokens of the sequence,
# so a 512 or 1024 cut-off would delete the label (train) or the question (inference cannot truncate at all).
# 2304 covers every row in train, dev and test, so nothing is ever truncated.
assert max(lens(clean).max(), lens(dev).max(), lens(test).max()) <= MAX_LEN
""")
code("""
def load_model(four_bit=False):
    kw = {"device_map": {"": 0}}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        kw["torch_dtype"] = torch.float16
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)

@torch.no_grad()
def predict(model, rows, batch_size=16, system_extra=""):
    \"\"\"Greedy decoding, max_new_tokens=200, output = generated text with whitespace stripped. Nothing else.\"\"\"
    model.eval()
    texts = [prompt_text(r, system_extra) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(texts[i]))        # longest first, less padding
    out = [None] * len(rows)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, do_sample=False, max_new_tokens=200, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen):
            out[i] = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return [{"id": r["id"], "output": o} for r, o in zip(rows, out)]

def evaluate(preds, gold=dev):
    res = score.score(gold, preds)
    return res, {"exact": round(res["all"]["exact_match"], 4), "field_acc": round(res["all"]["mean_field_acc"], 4),
                 "json_valid": round(res["all"]["json_valid"], 4)}

RESULTS = json.load(open("results.json")) if os.path.exists("results.json") else {}
def save_results():
    json.dump(RESULTS, open("results.json", "w"), indent=1)
""")
md("""
## Part B — Baselines (no fine-tuning)
1. base model, fixed prompt;  2. base model with the whole SCHEMA.md appended to the system prompt (reference only).
""")
code("""
if "baseline_fixed_prompt" not in RESULTS or "baseline_rules_in_prompt" not in RESULTS:
    base = load_model()
    rules_text = "\\n\\nFollow this labelling spec exactly. Reply with only the JSON object.\\n\\n" + open("SCHEMA.md", encoding="utf-8").read()
    for name, extra, bs in (("baseline_fixed_prompt", "", 16), ("baseline_rules_in_prompt", rules_text, 8)):
        t = time.time(); preds = predict(base, dev, bs, extra)
        _, RESULTS[name] = evaluate(preds); RESULTS[name]["sample"] = preds[0]["output"][:300]
        print(name, RESULTS[name], f"{(time.time() - t) / 60:.1f} min")
    save_results(); del base; gc.collect(); torch.cuda.empty_cache()
for k in ("baseline_fixed_prompt", "baseline_rules_in_prompt"): print(k, RESULTS[k])
""")
md("""
## Training loop

Main settings and why:

| Setting | Value | Reason |
|---|---|---|
| LoRA targets | all attention + MLP projections | the task needs new *computation* (date arithmetic, paise conversion, priority rules), not just a new output style; MLP adapters help with that |
| rank / alpha | 16 / 32 | 18.5M trainable params (limit 50M); rank is one of the ablations |
| learning rate | 2e-4, cosine, 5% warm-up | standard LoRA rate; adapters start at zero so a high rate is safe |
| epochs | 3 | ~1.3k rows; loss is flat by epoch 3 (see log) |
| batching | up to 16 rows, at most 3,000 padded tokens per batch | most tickets are ~200 tokens but threaded emails are ~1,700; a token budget keeps peak memory flat without truncating anything |
| precision | fp16 autocast, LoRA weights in fp32 | T4 has no fast bf16 |
""")
code("""
def make_batches(data, rng, max_rows=16, max_tokens=3000):
    idx = list(range(len(data))); rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), 256):                       # sort inside chunks: similar lengths share a batch
        chunk = sorted(idx[s:s + 256], key=lambda i: len(data[i]["input_ids"]))
        cur = []
        for i in chunk:
            longest = len(data[i]["input_ids"])
            if cur and (len(cur) == max_rows or (len(cur) + 1) * longest > max_tokens):
                batches.append(cur); cur = []
            cur.append(i)
        if cur: batches.append(cur)
    rng.shuffle(batches)
    return batches

def collate(items):
    n = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), n), tok.pad_token_id); lab = torch.full((len(items), n), -100); att = torch.zeros((len(items), n), dtype=torch.long)
    for j, x in enumerate(items):
        k = len(x["input_ids"])
        ids[j, :k] = torch.tensor(x["input_ids"]); lab[j, :k] = torch.tensor(x["labels"]); att[j, :k] = 1
    return ids.cuda(), att.cuda(), lab.cuda()

def run(name, rows, four_bit=False, r=16, lr=2e-4, epochs=3, note=""):
    if name in RESULTS:                                      # finished earlier in this session: reuse the saved log
        print(name, "(cached)", RESULTS[name]); return
    random.seed(SEED); torch.manual_seed(SEED); rng = random.Random(SEED)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = load_model(four_bit)
    if four_bit: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data = [encode(x) for x in rows]
    data = [x for x in data if len(x["input_ids"]) <= MAX_LEN]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = sum(len(make_batches(data, random.Random(e))) for e in range(epochs))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    model.train(); t0 = time.time(); step = 0; losses = []
    for ep in range(epochs):
        for b in make_batches(data, rng):
            ids, att, lab = collate([data[i] for i in b])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            losses.append(loss.item()); step += 1
            if step % 40 == 0: print(f"  {name} epoch {ep + 1} step {step}/{steps} loss {np.mean(losses[-40:]):.4f}")
    minutes = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    model.save_pretrained(f"runs/{name}")
    preds = predict(model, dev)
    json.dump(preds, open(f"runs/{name}/dev_preds.json", "w"))
    _, m = evaluate(preds)
    RESULTS[name] = dict(m, note=note, rows=len(data), trainable_params=trainable, peak_gpu_gb=round(peak, 2),
                         train_min=round(minutes, 1), final_loss=round(float(np.mean(losses[-20:])), 4))
    save_results(); print(name, RESULTS[name])
    del model, opt; gc.collect(); torch.cuda.empty_cache()
""")
md("""
## Part D — Controlled experiments
`A` is the reference run. Every other run changes exactly one thing relative to `A`.
""")
embed("augment.py")
code("""
import augment
aug = augment.augment(clean, n=500, seed=0)
print(len(clean), "clean rows +", len(aug), "synthetic rows. Examples:")
for r in aug[:3]: print("\\n" + r["messages"][1]["content"].split("\\n\\n", 1)[1].split("\\n>")[0][:300], "\\n ->", r["messages"][2]["content"])
""")
code("""
run("A_clean_r16_16bit", clean, note="reference: cleaned data, 16-bit LoRA, r=16, lr 2e-4, 3 epochs")
run("B_raw_data", raw_train, note="exp 1: raw train.jsonl instead of cleaned")
run("C_qlora_4bit", clean, four_bit=True, note="exp 2: base model in 4-bit NF4 (QLoRA) instead of 16-bit")
run("D_rank4", clean, r=4, note="exp 3: LoRA rank 4 instead of 16")
run("E_clean_plus_aug", clean + aug, note="exp 4 (own idea): + 500 rule-targeted synthetic rows")
""")
code("""
import pandas as pd
tab = pd.DataFrame(RESULTS).T[["note", "exact", "field_acc", "json_valid", "peak_gpu_gb", "train_min", "rows", "trainable_params"]]
tab
""")
md("""
## Final model, Part E error analysis, test predictions

The final model is the run with the best dev exact match. It is reloaded **from the saved adapter** onto a fresh
16-bit base model and decoded **one ticket at a time** — the same way the graders will run it — for both the final
dev numbers and the test predictions.
""")
code("""
runs = [k for k in RESULTS if not k.startswith("baseline") and k != "FINAL"]
best = max(runs, key=lambda k: (RESULTS[k]["exact"], RESULTS[k]["field_acc"]))
print("best run:", best)
shutil.rmtree("adapter", ignore_errors=True); shutil.copytree(f"runs/{best}", "adapter", ignore=shutil.ignore_patterns("dev_preds.json"))
final = PeftModel.from_pretrained(load_model(four_bit=(best == "C_qlora_4bit")), "adapter")
dev_preds = predict(final, dev, batch_size=1)
res, m = evaluate(dev_preds)
RESULTS["FINAL"] = dict(m, note=f"{best} reloaded from adapter/, batch size 1"); save_results()
with open("predictions_dev.jsonl", "w", encoding="utf-8") as f:
    for p in dev_preds: f.write(json.dumps(p, ensure_ascii=False) + "\\n")
!python score.py --gold data/dev.jsonl --pred predictions_dev.jsonl
""")
code("""
# Error analysis: every dev failure, with the fields that are wrong
gold = {r["id"]: json.loads(r["messages"][2]["content"]) for r in dev}
text = {r["id"]: r["messages"][1]["content"] for r in dev}
fails, field_err = [], collections.Counter()
for p in dev_preds:
    g = gold[p["id"]]; o = score.parse(p["output"]) or {}
    wrong = [k for k in score.KEYS if not (k in o and score.same(o[k], g[k]))]
    if wrong: fails.append((p["id"], wrong, o)); field_err.update(wrong)
print(len(fails), "of", len(dev), "dev tickets wrong. Errors per field:", dict(field_err.most_common()))
thread = {i for i, t in text.items() if "\\n>" in t}
for name, ids in (("email with quoted thread", thread), ("Hinglish", {i for i, g in gold.items() if g["language"] == "hi-en"}), ("English", {i for i, g in gold.items() if g["language"] == "en"})):
    bad = sum(f[0] in ids for f in fails); print(f"  slice {name:26} n={len(ids):3d} exact={1 - bad / max(len(ids), 1):.1%}")
for i, wrong, o in fails[:25]:
    print("\\n" + "=" * 100 + f"\\n{i}  wrong: " + ", ".join(f"{k}: pred={o.get(k)!r} gold={gold[i][k]!r}" for k in wrong))
    print(text[i].split("\\n>")[0].strip()[:600])
""")
code("""
t = time.time()
test_preds = predict(final, test, batch_size=1)
with open("predictions_test.jsonl", "w", encoding="utf-8") as f:
    for p in test_preds: f.write(json.dumps(p, ensure_ascii=False) + "\\n")
valid = sum(score.parse(p["output"]) is not None and list(score.parse(p["output"])) == score.KEYS for p in test_preds)
print(len(test_preds), "test predictions,", valid, "valid JSON with the 8 keys,", f"{(time.time() - t) / 60:.1f} min")
print(test_preds[0])
""")
code("""
# Reproducibility self-check: fresh base model + adapter/ from disk, 20 random test rows, must match the file exactly
del final; gc.collect(); torch.cuda.empty_cache()
check = PeftModel.from_pretrained(load_model(four_bit=(best == "C_qlora_4bit")), "adapter")
sample = random.Random(1).sample(range(len(test)), 20)
again = predict(check, [test[i] for i in sample], batch_size=1)
same = sum(a["output"] == test_preds[i]["output"] for a, i in zip(again, sample))
print(f"{same}/20 regenerated predictions identical")
print(sorted(os.listdir("adapter")), f"{os.path.getsize('adapter/adapter_model.safetensors') / 2**20:.0f} MB")
""")
code("""
# Bundle everything that goes into the repository
with zipfile.ZipFile("submission_outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["cleaning_log.csv", "predictions_test.jsonl", "predictions_dev.jsonl", "results.json"] + [f"adapter/{x}" for x in os.listdir("adapter")]:
        z.write(f)
    for k in runs: z.write(f"runs/{k}/dev_preds.json")
print(f"{os.path.getsize('submission_outputs.zip') / 2**20:.0f} MB")
try:
    from google.colab import files; files.download("submission_outputs.zip")
except Exception as e: print(e)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("notebook.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("notebook.ipynb written with", len(cells), "cells")
