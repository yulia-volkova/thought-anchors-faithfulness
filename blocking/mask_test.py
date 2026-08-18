"""Verify what attention_mask reaches custom vs standard attention kernels.

1) qwen2 14B with registered custom kernel (as extraction ran): print
   whether attention_mask is None at each call during a prefill forward.
2) gemma-3-12b with standard eager (as hiring attention ran): hook the
   first attention module and report whether the model materializes a mask.
Short prompts only; no generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

SEEN = {"none": 0, "mask": 0}


def spy_eager(module, query, key, value, attention_mask, scaling=None,
              dropout=0.0, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    if attention_mask is None:
        SEEN["none"] += 1
    else:
        SEEN["mask"] += 1
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return (torch.matmul(w, value_states).transpose(1, 2).contiguous(), None)


ALL_ATTENTION_FUNCTIONS["spy_eager"] = spy_eager

print("=== qwen2 14B, custom registered kernel (extraction path) ===")
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-14b")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-r1-distill-qwen-14b", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="spy_eager")
model.eval()
ids = tok("The capital of France is", return_tensors="pt")["input_ids"].to("cuda")
with torch.no_grad():
    out_custom = model(ids, output_hidden_states=True)
print(f"custom kernel calls: mask present {SEEN['mask']}, None {SEEN['none']}")

h_custom = out_custom.hidden_states[24][0, 2, :].float().cpu()
del out_custom
model.set_attn_implementation("sdpa")
with torch.no_grad():
    out_sdpa = model(ids, output_hidden_states=True)
h_sdpa = out_sdpa.hidden_states[24][0, 2, :].float().cpu()
cos = float((h_custom @ h_sdpa) /
            (h_custom.norm() * h_sdpa.norm()))
print(f"mid-position activation cos(custom, sdpa): {cos:.6f} "
      f"(1.0 = causality intact in custom path)")
del model, out_sdpa
torch.cuda.empty_cache()

print("=== gemma-3-12b, standard eager (hiring path) ===")
GS = {"none": 0, "mask": 0}
import transformers.models.gemma3.modeling_gemma3 as g3
orig = g3.eager_attention_forward


def spy_gemma(module, query, key, value, attention_mask, *a, **kw):
    if attention_mask is None:
        GS["none"] += 1
    else:
        GS["mask"] += 1
    return orig(module, query, key, value, attention_mask, *a, **kw)


g3.eager_attention_forward = spy_gemma
tok2 = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
m2 = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it", torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager")
m2.eval()
ids2 = tok2("The capital of France is", return_tensors="pt")["input_ids"].to("cuda")
with torch.no_grad():
    m2(ids2)
print(f"gemma eager calls: mask present {GS['mask']}, None {GS['none']}")
