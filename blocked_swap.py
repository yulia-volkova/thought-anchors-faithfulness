"""Cue-swap control for the attention-blocking experiment.

The 6 problems where following survived blocking are ambiguous: blocked
output should be mathematically independent of cue content (all post-cue
queries are cut off from cue-token columns at every layer, prefill
included), so surviving "influence" may be the blocked configuration's own
answer preference (the block also removes the attention-sink tokens), not
cue influence. Control: rerun blocked generation with the cue pointing to a
DIFFERENT wrong answer, same cue length. If rollouts still pick the
ORIGINAL cue answer, the survival is a blocking artifact; if they follow
the new cue, something leaks cue content past the block.

Also runs 3 problems whose influence collapsed under blocking, as controls.
"""

import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

MODEL = "deepseek-ai/deepseek-r1-distill-qwen-14b"
CUE_END = {"t": 0}
SURVIVORS = [30059, 30488, 30849, 30946, 40125, 40380]
N = 8


def blocked_eager(module, query, key, value, attention_mask, scaling=None,
                  dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    ce = CUE_END["t"]
    if ce > 0 and key_states.shape[-2] > ce:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        q_pos_start = k_len - q_len
        rows_after = max(0, ce - q_pos_start)
        scores[:, :, rows_after:, :ce] = float("-inf")
    w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, value_states)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS["blocked_eager"] = blocked_eager
ANS = re.compile(r"best answer is:?\s*\(?([A-J])\)?", re.I)


def main():
    crit = pd.read_csv("screening2_criteria.csv")
    sel = crit[((crit.gap >= 0.7) & (crit.nV >= 3) & (crit.nS >= 3)) | crit.ok]
    collapsed = [int(p) for p in sel.pi if int(p) not in SURVIVORS][:3]
    pis = SURVIVORS + collapsed
    prompts = pd.read_csv("data/screening_set2.csv")
    sub = prompts[prompts.pi.isin(pis)].set_index("pi")

    tokz = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="blocked_eager")
    model.eval()

    rows = []
    for pi in pis:
        r = sub.loc[pi]
        orig_cue, gt = r.cue_answer, r.gt_answer
        new_cue = next(c for c in "ABCD" if c not in (orig_cue, gt))
        old_line = r.prompt_cued.split("\n")[0]
        new_line = old_line.replace(f"({orig_cue})", f"({new_cue})")
        assert new_line != old_line, f"pi {pi}: cue letter not found in line"
        prompt = new_line + r.prompt_cued[len(old_line):]

        cue_txt = new_line + "\n"
        ce_ids = tokz(cue_txt, return_tensors="pt",
                      add_special_tokens=True)["input_ids"]
        full = tokz(prompt, return_tensors="pt",
                    add_special_tokens=True)["input_ids"]
        assert full[0, :ce_ids.shape[1]].tolist() == ce_ids[0].tolist(), \
            f"pi {pi}: boundary mismatch after swap"
        CUE_END["t"] = ce_ids.shape[1]

        ids = full.to("cuda")
        with torch.no_grad():
            out = model.generate(ids, do_sample=True, temperature=0.7,
                                 top_p=0.95, num_return_sequences=N,
                                 max_new_tokens=4096,
                                 pad_token_id=tokz.eos_token_id)
        f_orig = f_new = n_ans = 0
        for j in range(out.shape[0]):
            text = tokz.decode(out[j, ids.shape[1]:], skip_special_tokens=True)
            a = ANS.findall(text)
            ans = a[-1].upper() if a else None
            rows.append(dict(pi=int(pi), ridx=j, orig_cue=orig_cue,
                             new_cue=new_cue, gt=gt, answer=ans,
                             survivor=int(pi) in SURVIVORS))
            if ans:
                n_ans += 1
                f_orig += ans == orig_cue
                f_new += ans == new_cue
        pd.DataFrame(rows).to_parquet("blocked_swap.parquet", index=False)
        print(f"pi {pi} ({'survivor' if int(pi) in SURVIVORS else 'collapsed'}): "
              f"follow-ORIG {f_orig}/{n_ans}, follow-NEW {f_new}/{n_ans}",
              flush=True)
    print("done")


if __name__ == "__main__":
    main()
