"""Attention + activation extraction for the within-problem experiment.

For each problem in selected_mixed_problems.json, takes PER_CLASS verbalizing
and PER_CLASS silent stage-1 rollouts (14B-generated) and runs a teacher-forced
forward pass through the generating model. Per-layer forward hooks aggregate
token-level attention to sentence-level on the GPU and immediately free the
token-level tensor, so peak attention memory is one layer, not all layers
(8k-token GPQA sequences make output_attentions infeasible).

Saves per rollout (npz):
- verts: [n_layers, n_heads, n_sentences] raw vert scores
  (tril mean over rows >= i + proximity_ignore, drop_first NaN-ed)
- acts: [n_act_layers, 2, hidden] residual stream at last prompt token and
  last sequence token
- sentence metadata (prompt_len, n_sentences, label)

Usage:
  python extract_within_problem.py --limit 2 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   # smoke test
  python extract_within_problem.py                                                               # full 14B run
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from anchors_utils import (
    get_chunk_ranges,
    split_prompt_into_chunks,
    split_solution_into_chunks,
)
from select_mixed_problems import SOURCES, load_source, strip_prompt

DEFAULT_MODEL = "deepseek-ai/deepseek-r1-distill-qwen-14b"
PER_CLASS = 100  # take all available V/S rollouts per problem
PROXIMITY_IGNORE = 3
DROP_FIRST = 1
ACT_N_LAYERS = 16  # evenly spaced layers for activation saving


def build_token_ranges(full_text, prompt_char_len, tokenizer):
    """Sentence split prompt and generation, map to token ranges via offsets."""
    prompt_text = full_text[:prompt_char_len]
    gen_text = full_text[prompt_char_len:]

    prompt_sentences = split_prompt_into_chunks(prompt_text)
    gen_sentences = split_solution_into_chunks(gen_text)

    # Review F3 fix: tokenize prompt and generation SEPARATELY so the prompt's
    # token ids (and hence the pre-CoT boundary state) are identical across
    # continuations of the same prompt. Joint tokenization let the first
    # continuation characters alter the boundary token.
    enc_p = tokenizer(prompt_text, return_offsets_mapping=True,
                      return_tensors="pt", add_special_tokens=True)
    enc_g = tokenizer(gen_text, return_offsets_mapping=True,
                      return_tensors="pt", add_special_tokens=False)
    n_prompt_toks = enc_p["input_ids"].shape[1]

    import torch as _t
    full_input_ids = _t.cat([enc_p["input_ids"], enc_g["input_ids"]], dim=1)

    def ranges_for(offsets, char_ranges, shift):
        out = []
        for cs, ce in char_ranges:
            toks = [i for i, (ts, te) in enumerate(offsets)
                    if ts < ce and te > cs and te > ts]
            out.append((min(toks) + shift, max(toks) + 1 + shift) if toks else None)
        return out

    token_ranges = ranges_for(enc_p["offset_mapping"][0].tolist(),
                              get_chunk_ranges(prompt_text, prompt_sentences), 0)
    token_ranges += ranges_for(enc_g["offset_mapping"][0].tolist(),
                               get_chunk_ranges(gen_text, gen_sentences),
                               n_prompt_toks)
    prompt_last_tok = n_prompt_toks - 1

    sentences = prompt_sentences + gen_sentences
    keep = [i for i, tr in enumerate(token_ranges) if tr is not None]
    sentences = [sentences[i] for i in keep]
    token_ranges = [token_ranges[i] for i in keep]
    prompt_len = sum(1 for i in keep if i < len(prompt_sentences))
    return full_input_ids, sentences, token_ranges, prompt_len, prompt_last_tok


def make_pool_matrix(token_ranges, seq_len, device, dtype):
    """[T, S] matrix averaging tokens within each sentence range."""
    S = len(token_ranges)
    M = torch.zeros(seq_len, S, device=device, dtype=dtype)
    for j, (s, e) in enumerate(token_ranges):
        e = min(e, seq_len)
        if e > s:
            M[s:e, j] = 1.0 / (e - s)
    return M


class AttnCollector:
    """Holds the current pooling matrix and collected sentence matrices."""

    def __init__(self):
        self.pool = None  # [T, S] fp32, set per rollout
        self.sent_mats = {}
        self.ent_mats = {}

    def reset(self, pool):
        self.pool = pool
        self.sent_mats = {}
        self.ent_mats = {}


COLLECTOR = AttnCollector()
HEAD_CHUNK = 8


def pooled_eager_attention(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
    """Eager attention that pools weights to sentence level per head chunk.

    Standard eager attention materializes fp32 softmax over [H, T, T] per
    layer (~24 GB at T=8.5k), which OOMs alongside 14B weights. This variant
    computes softmax and the sentence pooling in chunks of HEAD_CHUNK heads,
    so full token-level attention never exists at once.
    """
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    n_heads = query.shape[1]
    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : key_states.shape[-2]]

    outs, sents, ents = [], [], []
    pool = COLLECTOR.pool
    T = query.shape[2]
    # normalizer: max possible entropy at each query position = log(context)
    ent_norm = torch.log(torch.arange(1, T + 1, device=query.device,
                                      dtype=torch.float32)).clamp(min=1.0)
    for h0 in range(0, n_heads, HEAD_CHUNK):
        scores = torch.matmul(
            query[:, h0:h0 + HEAD_CHUNK],
            key_states[:, h0:h0 + HEAD_CHUNK].transpose(2, 3)) * scaling
        if mask is not None:
            scores = scores + mask
        w = torch.softmax(scores, dim=-1, dtype=torch.float32)
        del scores
        outs.append(torch.matmul(w.to(query.dtype),
                                 value_states[:, h0:h0 + HEAD_CHUNK]))
        if pool is not None:
            tmp = w[0] @ pool                 # [ch, T, S] fp32
            sents.append((pool.T @ tmp).cpu())  # [ch, S, S]
            del tmp
            # normalized attention entropy per query token, then per sentence:
            # length-invariant diffuseness (1 = uniform, 0 = one-hot)
            row_ent = -(w[0] * torch.log(w[0] + 1e-12)).sum(-1)  # [ch, T]
            row_ent = row_ent / ent_norm
            ents.append((row_ent @ pool).cpu())  # [ch, S] mean over sentence tokens
            del row_ent
        del w
    attn_output = torch.cat(outs, dim=1).transpose(1, 2).contiguous()
    if sents:
        COLLECTOR.sent_mats[module.layer_idx] = torch.cat(sents)
        COLLECTOR.ent_mats[module.layer_idx] = torch.cat(ents)
    return attn_output, None


def vert_scores_from_sent_mats(sent_mats, n_layers, n_heads):
    """[n_layers, n_heads, S] raw vert scores (mirrors get_attn_vert_scores,
    rank_normalize=False)."""
    S = next(iter(sent_mats.values())).shape[-1]
    verts = np.full((n_layers, n_heads, S), np.nan, dtype=np.float32)
    for li, sent in sent_mats.items():
        m = torch.tril(sent).numpy()  # [H, S, S]
        for i in range(S):
            rows = m[:, i + PROXIMITY_IGNORE:, i]
            verts[li, :, i] = rows.mean(axis=1) if rows.shape[1] > 0 else np.nan
    verts[:, :, :DROP_FIRST] = np.nan
    verts[:, :, -DROP_FIRST:] = np.nan
    return verts


def pick_rollouts(sel_entry):
    rng = np.random.default_rng(0)
    v = sel_entry["verbalizing_idx"]
    s = sel_entry["silent_idx"]
    v = list(rng.permutation(v)[:PER_CLASS])
    s = list(rng.permutation(s)[:PER_CLASS])
    return [(int(i), "verbalizing") for i in v] + [(int(i), "silent") for i in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default="extracted_within_problem")
    ap.add_argument("--limit", type=int, default=None, help="max problems (smoke test)")
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "gpqa"])
    args = ap.parse_args()

    with open("selected_mixed_problems.json") as f:
        selection = json.load(f)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    # fp16 overflows in these distills' forward pass (NaN from layer 1 on);
    # they are bf16-native. Pooling math stays fp32 via make_pool_matrix.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["pooled_eager"] = pooled_eager_attention
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        attn_implementation="pooled_eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    act_layers = sorted(set(np.linspace(0, n_layers - 1, ACT_N_LAYERS, dtype=int).tolist()))

    os.makedirs(args.out_dir, exist_ok=True)

    for ds in args.datasets:
        df = load_source(SOURCES[ds])
        pi_col = "pi" if "pi" in df.columns else "qid"
        if "response_idx" not in df.columns:
            df["response_idx"] = df.groupby(pi_col).cumcount()
        entries = selection[ds][: args.limit] if args.limit else selection[ds]

        for entry in entries:
            pi = entry["pi"]
            g = df[df[pi_col] == pi]
            for ridx, label in pick_rollouts(entry):
                out_path = os.path.join(args.out_dir, f"{ds}_{pi}_{ridx}.npz")
                if os.path.exists(out_path):
                    continue
                row = g[g["response_idx"] == ridx].iloc[0]
                gen = strip_prompt(row)
                prompt = row["question_with_cue"]
                full_text = prompt + gen

                input_ids, sentences, token_ranges, prompt_len, prompt_last_tok = \
                    build_token_ranges(full_text, len(prompt), tokenizer)
                T = input_ids.shape[1]
                if T > 16384:
                    print(f"  skip {ds} pi {pi} r{ridx}: {T} tokens")
                    continue

                pool = make_pool_matrix(token_ranges, T, device, torch.float32)
                COLLECTOR.reset(pool)
                act_store = {}

                handles = []

                def act_hook(li):
                    def fn(module, args_, output):
                        h = output[0] if isinstance(output, tuple) else output
                        act_store[li] = h[0, [prompt_last_tok, -1], :].float().cpu()
                    return fn
                handles += [
                    model.model.layers[li].register_forward_hook(act_hook(li))
                    for li in act_layers
                ]

                with torch.no_grad():
                    model(input_ids.to(device), use_cache=False)
                for h in handles:
                    h.remove()

                verts = vert_scores_from_sent_mats(
                    COLLECTOR.sent_mats, n_layers, n_heads)
                acts = np.stack([act_store[li].numpy() for li in act_layers])

                ents = np.stack([
                    COLLECTOR.ent_mats[li].numpy()
                    for li in sorted(COLLECTOR.ent_mats)
                ])  # [L, H, S] normalized entropy per sentence
                np.savez_compressed(
                    out_path,
                    verts=verts,
                    ents=ents,
                    acts=acts,
                    act_layers=np.array(act_layers),
                    prompt_len=prompt_len,
                    n_sentences=len(sentences),
                    n_tokens=T,
                    label=label,
                )
                print(f"  {ds} pi {pi} r{ridx} [{label}]: {T} tokens, "
                      f"{len(sentences)} sentences -> {out_path}")
                COLLECTOR.reset(None); del pool
                if device == "cuda":
                    torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()
