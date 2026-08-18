"""Targeted-attention test on the Karvonen & Marks hiring setting.

For each logged record (resume with name variant, gemma-3-12b-it), run a
teacher-forced forward pass of the exact logged prompt and measure attention
from the final (decision) position to the name's token span, per head, on
GLOBAL attention layers only (gemma-3 local layers cannot reach distant
tokens by construction).

Question: does attention-to-name predict name-sensitivity of the decision
(delta p(yes) across the resume's 4 name variants)?

Usage:
  python hiring_attention.py --pkl <file.pkl> [--limit 8] --out-dir hiring_out
"""

import argparse
import glob
import os
import pickle

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-3-12b-it"


COLLEGES = ["Howard University", "Morehouse College", "Georgetown University", "Emory University"]


def name_token_span(prompt, name, tokenizer, enc_offsets):
    # college mode: span = the college mention, not the person name
    import os
    if os.environ.get("SPAN_MODE") == "college":
        for c in COLLEGES:
            if c in prompt:
                name = c
                break
        else:
            return None
    start = prompt.find(name)
    if start < 0:
        return None
    end = start + len(name)
    toks = [i for i, (ts, te) in enumerate(enc_offsets)
            if ts < end and te > start and te > ts]
    return (min(toks), max(toks) + 1) if toks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, nargs="+")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="hiring_out")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    # global layers: gemma-3 uses sliding window except every Nth layer
    pattern = getattr(cfg, "sliding_window_pattern", 6)
    n_layers = cfg.num_hidden_layers
    global_layers = [i for i in range(n_layers) if (i + 1) % pattern == 0]
    print(f"{n_layers} layers, global: {global_layers}")

    os.makedirs(args.out_dir, exist_ok=True)
    for pkl_path in args.pkl:
        with open(pkl_path, "rb") as fh:
            d = pickle.load(fh)
        res = d["results"][: args.limit] if args.limit else d["results"]
        base = os.path.basename(pkl_path).replace(".pkl", "")
        out_rows = []
        for i, r in enumerate(res):
            prompt = r["prompt"]
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_offsets_mapping=True,
                            return_tensors="pt", add_special_tokens=False)
            span = name_token_span(text, r["name"], tokenizer,
                                   enc["offset_mapping"][0].tolist())
            if span is None:
                print(f"  [{i}] name not found, skip")
                continue
            if r["yes_probs"] is None:
                print(f"  [{i}] yes_probs is None, skip")
                continue
            with torch.no_grad():
                out = model(enc["input_ids"].to("cuda"), use_cache=False,
                            output_attentions=True)
            attn_to_name = {}
            attn_profile = {}
            for li in global_layers:
                A = out.attentions[li][0].float()  # [H, T, T]
                attn_to_name[li] = A[:, -1, span[0]:span[1]].sum(-1).cpu().numpy()
                # received-by-name profile: from every query position, per head
                attn_profile[li] = A[:, :, span[0]:span[1]].sum(-1).half().cpu().numpy()  # [H, T]
            del out
            torch.cuda.empty_cache()
            out_rows.append(dict(
                idx=i, name=r["name"], race=r["race"], gender=r["gender"],
                yes_probs=float(r["yes_probs"]),
                n_tokens=int(enc["input_ids"].shape[1]),
                span=span,
                attn_to_name={str(k): v.tolist() for k, v in attn_to_name.items()},
                attn_profile={str(k): v for k, v in attn_profile.items()},
            ))
            if i % 50 == 0:
                print(f"  {base}: {i}/{len(res)}")
        with open(os.path.join(args.out_dir, base + "_attn.pkl"), "wb") as fh:
            pickle.dump(out_rows, fh)
        print(f"{base}: saved {len(out_rows)} records")


if __name__ == "__main__":
    main()
