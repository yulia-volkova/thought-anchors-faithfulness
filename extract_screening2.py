"""Attention+activation extraction for the screening-2 union set.

Reads extraction_manifest.parquet (ds, pi, ridx, label, prompt_cued, text) and
reuses the fixed machinery from extract_within_problem (pooled attention kernel,
separate prompt/generation tokenization, entropy, activations).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import extract_within_problem as E

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=E.DEFAULT_MODEL)
ap.add_argument("--out-dir", default="extracted_s2")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

m = pd.read_parquet("extraction_manifest.parquet")
if args.limit:
    m = m.head(args.limit)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ALL_ATTENTION_FUNCTIONS["pooled_eager"] = E.pooled_eager_attention
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=dtype, device_map=device,
    attn_implementation="pooled_eager")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model)
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
act_layers = sorted(set(np.linspace(0, n_layers - 1, E.ACT_N_LAYERS, dtype=int).tolist()))
os.makedirs(args.out_dir, exist_ok=True)

for _, r in m.iterrows():
    out_path = os.path.join(args.out_dir, f"{r.ds}_{r.pi}_{r.ridx}.npz")
    if os.path.exists(out_path):
        continue
    full_text = r.prompt_cued + str(r.text)
    input_ids, sentences, token_ranges, prompt_len, prompt_last_tok = \
        E.build_token_ranges(full_text, len(r.prompt_cued), tokenizer)
    T = input_ids.shape[1]
    if T > 13000:
        print(f"skip {r.ds} {r.pi} r{r.ridx}: {T} tokens", flush=True)
        continue
    pool = E.make_pool_matrix(token_ranges, T, device, torch.float32)
    E.COLLECTOR.reset(pool)
    act_store = {}
    def act_hook(li):
        def fn(module, a, output):
            h = output[0] if isinstance(output, tuple) else output
            act_store[li] = h[0, [prompt_last_tok, -1], :].float().cpu()
        return fn
    handles = [model.model.layers[li].register_forward_hook(act_hook(li))
               for li in act_layers]
    with torch.no_grad():
        model(input_ids.to(device), use_cache=False)
    for h in handles:
        h.remove()
    verts = E.vert_scores_from_sent_mats(E.COLLECTOR.sent_mats, n_layers, n_heads)
    ents = np.stack([E.COLLECTOR.ent_mats[li].numpy()
                     for li in sorted(E.COLLECTOR.ent_mats)])
    acts = np.stack([act_store[li].numpy() for li in act_layers])
    np.savez_compressed(out_path, verts=verts, ents=ents, acts=acts,
                        act_layers=np.array(act_layers),
                        prompt_len=prompt_len, n_sentences=len(sentences),
                        n_tokens=T, label=str(r.label), gap=float(r.gap),
                        prereg=bool(r.prereg))
    print(f"{r.ds} {r.pi} r{r.ridx} [{r.label}] {T}tok {len(sentences)}sent", flush=True)
print("done")
