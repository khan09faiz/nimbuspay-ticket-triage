# NimbusPay ticket triage — LoRA fine-tune

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, Apache-2.0)
- **Loaded in 4-bit:** **No.** The final adapter was trained and evaluated on the base model in 16-bit (`torch_dtype=torch.float16`). Load it the same way.
- **Best result:** 95.0% exact match / 99.0% mean field accuracy on dev (200 tickets), from LoRA r=16 trained on the cleaned data plus 500 rule-targeted synthetic rows. Base model with the fixed prompt: 0%.

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | Colab notebook with outputs. Audit → baselines → 5 experiments → error analysis → test predictions. |
| `report.md` | Audit findings, experiment table, error analysis, next steps, AI-tool disclosure. |
| `adapter/` | Final LoRA adapter in PEFT format (18.5M trainable parameters, 74 MB). |
| `predictions_test.jsonl` | 400 raw model outputs for `test_inputs.jsonl`. |
| `cleaning_log.csv` | Every dropped or fixed training row: `id,action,reason` (171 drops, 175 fixes). |
| `predictions_dev.jsonl`, `results.json` | Dev outputs of the final model and the metrics of every run. |
| `rules.py`, `audit.py`, `augment.py` | Same code the notebook writes with `%%writefile`: the spec as code, the audit, the augmentation. Used on training data only. |
| `score.py` | The scorer from the candidate pack. |

## Loading the adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()

text = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)   # system + user, model's own template
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate(**enc, do_sample=False, max_new_tokens=200)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

Predictions in `predictions_test.jsonl` were generated exactly like this: one ticket per call, greedy, no post-processing.

## Reproducing

Open `notebook.ipynb` in Colab on a T4, put `training_assessment.zip` in the runtime (or upload when asked) and run all. The full run took roughly 2.5 hours on a free T4 (5 training runs of 13–17 min each, plus evaluation).
