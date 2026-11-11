# NimbusPay ticket triage: LoRA fine-tune

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, Apache-2.0)
- **Loaded in 4-bit:** No. I trained and evaluated the final adapter on the base model in 16-bit (`torch_dtype=torch.float16`). Please load it the same way.
- **Best result:** 95.0% exact match and 99.0% mean field accuracy on dev (200 tickets). This is LoRA r=16 trained on my cleaned data plus 500 synthetic rows. The base model with the fixed prompt gets 0%.

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | Colab notebook with outputs: audit, baselines, 5 experiments, error analysis, test predictions. |
| `report.md` | Audit findings, experiment table, error analysis, next steps, AI tools used. |
| `adapter/` | Final LoRA adapter in PEFT format (18.5M trainable parameters, 74 MB). |
| `predictions_test.jsonl` | 400 raw model outputs for `test_inputs.jsonl`. |
| `cleaning_log.csv` | Every training row I dropped or fixed, as `id,action,reason` (171 drops, 175 fixes). |
| `predictions_dev.jsonl`, `results.json` | Dev outputs of the final model and the metrics of every run. |
| `rules.py`, `audit.py`, `augment.py` | The same code the notebook writes with `%%writefile`: the spec as code, the audit and the augmentation. Used on training data only. |
| `score.py` | The scorer from the candidate pack. |

## Loading the adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()

# system + user message, model's own chat template
text = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate(**enc, do_sample=False, max_new_tokens=200)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

`predictions_test.jsonl` was generated this way: one ticket per call, greedy decoding, nothing done to the output except stripping whitespace.

## Reproducing

Open `notebook.ipynb` in Colab on a T4, put `training_assessment.zip` in the runtime (or upload it when the second cell asks) and run all cells. My full run took about 2.5 hours on a free T4. Each of the 5 training runs takes 13 to 17 minutes and the rest is evaluation.
