"""Attention-blocking causal experiment (Goodfire suggestion, pre-registered).

For each analyzed problem: generate cued rollouts with attention to the cue
sentence's tokens BLOCKED (scores set to -inf at those key positions for all
query positions after the cue; softmax renormalizes). Compare follow rates:
cued-normal (screening data) vs cued-blocked (here) vs uncued (screening data).

Usage: python blocked_generation.py [--limit N] [--n 8] [--out blocked_gen.parquet]
"""

import argparse
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

MODEL = "deepseek-ai/deepseek-r1-distill-qwen-14b"
CUE_END = {"t": 0}  # token index where cue sentence ends (set per problem)


def blocked_eager(module, query, key, value, attention_mask, scaling=None,
                  dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    else:
        # transformers passes None to custom-registered kernels; without an
        # explicit causal mask the forward is bidirectional
        Tq, Tk = scores.shape[-2], scores.shape[-1]
        cm = torch.full((Tq, Tk), torch.finfo(torch.float32).min,
                        device=scores.device, dtype=scores.dtype)
        scores = scores + torch.triu(cm, diagonal=Tk - Tq + 1)[None, None]
    ce = CUE_END["t"]
    if ce > 0 and key_states.shape[-2] > ce:
        # block attention TO cue tokens from all queries strictly after the cue
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        q_pos_start = k_len - q_len  # absolute position of first query row
        rows_after = max(0, ce - q_pos_start)  # rows representing pos >= ce
        scores[:, :, rows_after:, :ce] = float("-inf")
    w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, value_states)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS["blocked_eager"] = blocked_eager
ANS = re.compile(r"best answer is:?\s*\(?([A-J])\)?", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="blocked_gen.parquet")
    args = ap.parse_args()

    crit = pd.read_csv("screening2_criteria.csv")
    sel = crit[((crit.gap >= 0.7) & (crit.nV >= 3) & (crit.nS >= 3)) | crit.ok]
    prompts = pd.read_csv("data/screening_set2.csv")
    sel = sel.merge(prompts[["pi", "prompt_cued", "gt_answer", "cue_answer"]], on="pi")
    if args.limit:
        sel = sel.head(args.limit)

    tokz = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="blocked_eager")
    model.eval()

    rows = []
    for _, r in sel.iterrows():
        cue_txt = r.prompt_cued.split("\n")[0] + "\n"  # prepended professor line
        CUE_END["t"] = tokz(cue_txt, return_tensors="pt",
                            add_special_tokens=True)["input_ids"].shape[1]
        ids = tokz(r.prompt_cued, return_tensors="pt",
                   add_special_tokens=True)["input_ids"].to("cuda")
        with torch.no_grad():
            out = model.generate(ids, do_sample=True, temperature=0.7, top_p=0.95,
                                 num_return_sequences=args.n, max_new_tokens=4096,
                                 pad_token_id=tokz.eos_token_id)
        for j in range(out.shape[0]):
            text = tokz.decode(out[j, ids.shape[1]:], skip_special_tokens=True)
            a = ANS.findall(text)
            rows.append(dict(pi=int(r.pi), ridx=j, gap=float(r.gap),
                             gt=r.gt_answer, cue=r.cue_answer,
                             answer=a[-1].upper() if a else None, text=text))
        pd.DataFrame(rows).to_parquet(args.out, index=False)
        got = pd.DataFrame(rows)
        g = got[got.pi == r.pi]
        print(f"pi {r.pi}: blocked follow "
              f"{(g.answer == g.cue).mean():.2f} (gap was {r.gap:.2f})", flush=True)
    print("done")


if __name__ == "__main__":
    main()
