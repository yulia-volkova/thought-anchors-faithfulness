#!/usr/bin/env python3
"""
Attention analysis script for Chain-of-Thought faithfulness research.
Equivalent to anchors_unified.ipynb but without plotting.

This script:
1. Loads model and problems data
2. Generates rollouts for specified problem IDs (cued and uncued conditions)
3. Computes attention-based receiver heads using kurtosis of vertical scores
4. Saves all results for later visualization/analysis
5. Computes aggregate statistics across problem categories

Usage:
    python run_anchors_analysis.py --dataset mmlu
    python run_anchors_analysis.py --dataset gpqa --load-from-saved
    python run_anchors_analysis.py --dataset mmlu --pis 91 152 188 --force
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy import stats
import json
import re
import os
import gc
import argparse
import pickle
from collections import defaultdict
from datetime import datetime

# CUDA memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# =============================================================================
# Configuration
# =============================================================================

DATASET_CONFIG = {
    "gpqa": {
        "problems_file": "data/selected_problems_gpqa-8192-mt.json",
        "save_dir": "final/gpqa",
        "max_new_tokens": 2048,
        "faithful_pis": [162, 172, 129, 160, 21],
        "unfaithful_pis": [116, 101, 107, 100, 134],
    },
    "mmlu": {
        "problems_file": "data/selected_problems_mmlu.json",
        "save_dir": "final/mmlu",
        "max_new_tokens": 2048,
        "faithful_pis": [91, 152, 188],
        "unfaithful_pis": [19, 151, 182, 191],
    },
}

# Model settings
MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-14b"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis parameters
NUM_ROLLOUTS_PER_CONDITION = 5
TOP_K_RECEIVER_HEADS = 5
DROP_FIRST = 1
PROXIMITY_IGNORE = 3
INCLUDE_PROMPT = True
TEMPERATURE = 0.7
TOP_P = 0.95

# Cue patterns
CUE_PATTERNS = [
    r"professor", r"stanford", r"iq\s*(?:of)?\s*130"
]


# =============================================================================
# Imports from anchors_utils
# =============================================================================

from anchors_utils import split_solution_into_chunks, get_chunk_ranges, get_chunk_token_ranges, split_prompt_into_chunks


# =============================================================================
# Core Analysis Functions
# =============================================================================

def find_cue_sentences(sentences, patterns=CUE_PATTERNS):
    """Find sentence indices that mention cue patterns."""
    idxs = []
    for i, s in enumerate(sentences):
        s2 = s.lower()
        if any(re.search(pat, s2) for pat in patterns):
            idxs.append(i)
    return idxs


def avg_matrix_by_chunk(matrix, chunk_token_ranges):
    """Average attention matrix by chunk (sentence-level)."""
    n = len(chunk_token_ranges)
    avg_mat = np.zeros((n, n), dtype=np.float32)
    for i, (si, ei) in enumerate(chunk_token_ranges):
        for j, (sj, ej) in enumerate(chunk_token_ranges):
            region = matrix[si:ei, sj:ej]
            avg_mat[i, j] = region.mean().item() if region.size > 0 else np.nan
    return avg_mat


def rank_normalize_rows(matrix):
    """Rank-normalize each row of the matrix (matches original control_depth)."""
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        valid_mask = ~np.isnan(row)
        if valid_mask.sum() > 0:
            ranks = stats.rankdata(row[valid_mask], method='average')
            # Normalize to [0, 1]
            ranks = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.array([0.5])
            result[i, valid_mask] = ranks
        result[i, ~valid_mask] = np.nan
    return result


def get_attn_vert_scores(avg_mat, proximity_ignore=PROXIMITY_IGNORE, drop_first=DROP_FIRST, rank_normalize=True):
    """
    Compute vertical scores for each source sentence.

    Args:
        avg_mat: Sentence-level attention matrix
        proximity_ignore: Ignore this many nearby sentences
        drop_first: Set first/last N scores to NaN
        rank_normalize: Apply rank normalization per row (matches original control_depth)
    """
    avg_mat = np.tril(avg_mat.copy())  # later->earlier only

    if rank_normalize:
        avg_mat = rank_normalize_rows(avg_mat)

    n = avg_mat.shape[0]
    vert_scores = []
    for i in range(n):
        vert_lines = avg_mat[i + proximity_ignore:, i]
        vert_score = np.nanmean(vert_lines) if len(vert_lines) > 0 else np.nan
        vert_scores.append(vert_score)
    vert_scores = np.array(vert_scores)
    if drop_first > 0:
        vert_scores[:drop_first] = np.nan
        vert_scores[-drop_first:] = np.nan
    return vert_scores


def compute_kurtosis_per_head(rollout_vert_scores):
    """Compute kurtosis by pooling all scores from all rollouts (matches original)."""
    head2kurt = {}
    for lh, vs_list in rollout_vert_scores.items():
        # Pool all scores from all rollouts into one array
        vs_pooled = np.concatenate([vs for vs in vs_list if len(vs) > 0])
        if len(vs_pooled) < 4:  # Need enough points for kurtosis
            continue
        head2kurt[lh] = stats.kurtosis(vs_pooled, fisher=True, bias=True, nan_policy="omit")
    return head2kurt


def select_top_heads(head2kurt, top_k=TOP_K_RECEIVER_HEADS, head2verts=None, min_max_vert=0.001):
    """
    Select top heads by kurtosis, but filter out heads with negligible attention.

    Args:
        head2kurt: dict of (layer, head) -> kurtosis value
        top_k: number of top heads to return
        head2verts: dict of (layer, head) -> list of vertical score arrays (for filtering)
        min_max_vert: minimum max vertical score required (filters out inactive heads)
    """
    items = [(k, v) for k, v in head2kurt.items() if not np.isnan(v)]

    # Filter out heads with negligible attention if head2verts is provided
    if head2verts is not None:
        filtered_items = []
        for k, v in items:
            vs_list = head2verts.get(k, [])
            if vs_list:
                max_vert = max(np.nanmax(vs) if len(vs) > 0 else 0 for vs in vs_list)
                if max_vert >= min_max_vert:
                    filtered_items.append((k, v))
        items = filtered_items

    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_k]


# =============================================================================
# Model and Generation Functions
# =============================================================================

def load_model(model_name=MODEL_NAME):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


def get_attention_weights(model, tokenizer, input_ids):
    """Get attention weights for input_ids."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(model.device),
            output_attentions=True,
            use_cache=False,
        )
    return outputs.attentions


def run_rollouts(model, tokenizer, prompt, num_rollouts, max_new_tokens, include_prompt=True):
    """Generate multiple rollouts for a prompt."""
    rollouts = []

    for i in range(num_rollouts):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        prompt_len_tokens = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Get prompt text
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Split prompt into chunks
        prompt_sentences = split_prompt_into_chunks(prompt_text) if include_prompt else []
        prompt_ranges = get_chunk_ranges(prompt_text, prompt_sentences) if prompt_sentences else []
        prompt_token_ranges = get_chunk_token_ranges(prompt_text, prompt_ranges, tokenizer) if prompt_sentences else []
        prompt_len = len(prompt_sentences)

        # Split generation into chunks
        roll_sentences = split_solution_into_chunks(generated_text)
        roll_ranges = get_chunk_ranges(generated_text, roll_sentences)
        roll_token_ranges = get_chunk_token_ranges(generated_text, roll_ranges, tokenizer)

        if include_prompt:
            sentences = prompt_sentences + roll_sentences
            token_ranges = prompt_token_ranges + roll_token_ranges
        else:
            sentences = roll_sentences
            token_ranges = roll_token_ranges
            prompt_len = 0

        rollout = {
            "ids": generated_ids,
            "text": generated_text,
            "sentences": sentences,
            "token_ranges": token_ranges,
            "prompt_len": prompt_len,
            "prompt_len_tokens": prompt_len_tokens,
        }
        rollouts.append(rollout)
        print(f"    Rollout {i+1}/{num_rollouts}: {len(sentences)} sentences, {len(generated_ids)} tokens")

    return rollouts


def collect_vert_scores_for_rollouts(model, tokenizer, rollouts, cache_attention=False):
    """Collect vertical scores for all heads across all rollouts."""
    head2verts_full = defaultdict(list)
    head2verts_reasoning = defaultdict(list)
    cached_attentions = []

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    for ro_idx, ro in enumerate(rollouts):
        print(f"      Processing rollout {ro_idx + 1}/{len(rollouts)}...")

        # Get attention weights
        attn_weights = get_attention_weights(model, tokenizer, ro["ids"].unsqueeze(0))

        if cache_attention:
            # Cache as numpy to save memory
            cached = {f"L{l}": attn_weights[l][0].cpu().numpy() for l in range(n_layers)}
            cached_attentions.append(cached)

        token_ranges = ro["token_ranges"]
        prompt_len = ro["prompt_len"]

        for layer in range(n_layers):
            for head in range(n_heads):
                mat = attn_weights[layer][0, head].cpu().numpy()

                # Full (with prompt)
                avg_mat_full = avg_matrix_by_chunk(mat, token_ranges)
                vert_scores_full = get_attn_vert_scores(avg_mat_full, rank_normalize=True)
                head2verts_full[(layer, head)].append(vert_scores_full)

                # Reasoning only (exclude prompt)
                if prompt_len > 0 and prompt_len < len(token_ranges):
                    reasoning_ranges = token_ranges[prompt_len:]
                    avg_mat_reasoning = avg_matrix_by_chunk(mat, reasoning_ranges)
                    vert_scores_reasoning = get_attn_vert_scores(avg_mat_reasoning, rank_normalize=True)
                    head2verts_reasoning[(layer, head)].append(vert_scores_reasoning)

        # Clear GPU memory
        del attn_weights
        torch.cuda.empty_cache()

    return dict(head2verts_full), dict(head2verts_reasoning), cached_attentions if cache_attention else None


# =============================================================================
# Save/Load Functions
# =============================================================================

def get_save_path(pi, num_rollouts, top_k, drop_first, base_dir):
    """Generate save directory path."""
    return os.path.join(base_dir, f"{pi}_{num_rollouts}_{top_k}_{drop_first}")


def check_professor_mention(rollout):
    """Check if a rollout's response mentions professor/cue."""
    prompt_len = rollout["prompt_len"]
    response_sentences = rollout["sentences"][prompt_len:]
    response_text = " ".join(response_sentences).lower()
    return any(pat in response_text for pat in ["professor", "stanford", "iq of 130", "iq 130"])


def select_rollout_for_saving(rollouts, condition, cached_attentions=None):
    """Select one rollout for saving with attention."""
    if condition == "cued":
        # Prefer rollouts that mention professor
        for i, ro in enumerate(rollouts):
            if check_professor_mention(ro):
                return i, ro, cached_attentions[i] if cached_attentions else None
    # Default to first rollout
    return 0, rollouts[0], cached_attentions[0] if cached_attentions else None


def save_analysis_results(save_path, config, rollout_data, head2verts_data, top_heads_data, attention_data=None):
    """Save all analysis results."""
    os.makedirs(save_path, exist_ok=True)

    # Save config
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save rollout data (without tensors)
    for condition in ["cued", "uncued"]:
        ro = rollout_data[condition]
        ro_save = {
            "text": ro["text"],
            "sentences": ro["sentences"],
            "token_ranges": ro["token_ranges"],
            "prompt_len": ro["prompt_len"],
            "prompt_len_tokens": ro["prompt_len_tokens"],
        }
        with open(os.path.join(save_path, f"{condition}_rollout.json"), "w") as f:
            json.dump(ro_save, f, indent=2)

    # Save head2verts (serialized)
    for key, h2v in head2verts_data.items():
        serialized = {f"{l}_{h}": [vs.tolist() for vs in vs_list] for (l, h), vs_list in h2v.items()}
        with open(os.path.join(save_path, f"{key}.json"), "w") as f:
            json.dump(serialized, f)

    # Save top heads
    for key, heads in top_heads_data.items():
        serialized = [(list(h), float(k)) for h, k in heads]
        with open(os.path.join(save_path, f"{key}.json"), "w") as f:
            json.dump(serialized, f, indent=2)

    # Save attention data (numpy)
    if attention_data:
        for condition, attn in attention_data.items():
            if attn:
                np.savez_compressed(os.path.join(save_path, f"{condition}_attention.npz"), **attn)

    print(f"    Saved results to {save_path}")


def load_head2verts(pi, condition, reasoning_only, save_dir, num_rollouts, top_k, drop_first):
    """Load head2verts for a single PI."""
    suffix = "_reasoning" if reasoning_only else ""
    path = os.path.join(save_dir, f"{pi}_{num_rollouts}_{top_k}_{drop_first}",
                        f"{condition}_head2verts{suffix}.json")
    with open(path, "r") as f:
        data = json.load(f)
    h2v = {}
    for key, vs_list in data.items():
        l, h = map(int, key.split("_"))
        h2v[(l, h)] = [np.array(vs) for vs in vs_list]
    return h2v


# =============================================================================
# Aggregate Analysis Functions
# =============================================================================

def aggregate_kurtosis(pi_list, condition, reasoning_only, save_dir, num_rollouts, top_k, drop_first):
    """
    Compute aggregate kurtosis from multiple PIs.
    Pools all vertical scores from all rollouts from all PIs, then computes kurtosis once (matches original).
    """
    head2scores = defaultdict(list)
    mode_str = "reasoning-only" if reasoning_only else "full"

    for pi in pi_list:
        try:
            h2v = load_head2verts(pi, condition, reasoning_only, save_dir, num_rollouts, top_k, drop_first)
        except FileNotFoundError:
            print(f"   [!] PI {pi} ({mode_str}) not found, skipping")
            continue

        for (layer, head), vs_list in h2v.items():
            for vs in vs_list:
                if len(vs) > 0:
                    head2scores[(layer, head)].extend(vs.tolist())

    # Compute kurtosis on pooled scores
    head2kurt = {}
    for h, scores in head2scores.items():
        if len(scores) >= 4:
            k = stats.kurtosis(scores, fisher=True, bias=True, nan_policy="omit")
            if not np.isnan(k):
                head2kurt[h] = k

    return head2kurt, head2scores


def get_aggregate_top_heads(pi_list, condition, reasoning_only, save_dir, num_rollouts, top_k, drop_first):
    """Get top heads for a category by aggregating across PIs."""
    head2kurt, head2scores = aggregate_kurtosis(
        pi_list, condition, reasoning_only, save_dir, num_rollouts, top_k, drop_first
    )

    items = [(k, v) for k, v in head2kurt.items() if not np.isnan(v)]
    items.sort(key=lambda x: x[1], reverse=True)

    return items[:top_k], head2kurt, head2scores


def save_aggregate_results(save_dir, faithful_pis, unfaithful_pis, num_rollouts, top_k, drop_first):
    """Compute and save aggregate results for both categories."""
    aggregate_dir = os.path.join(save_dir, "aggregate")
    os.makedirs(os.path.join(aggregate_dir, "faithful"), exist_ok=True)
    os.makedirs(os.path.join(aggregate_dir, "unfaithful"), exist_ok=True)

    print("\nComputing aggregate top heads (FULL - with prompt)...")

    # Faithful
    print(f"\n{'='*50}")
    print(f"FAITHFUL (PIs: {faithful_pis})")
    print(f"{'='*50}")
    faithful_top_cued, _, _ = get_aggregate_top_heads(
        faithful_pis, "cued", False, save_dir, num_rollouts, top_k, drop_first
    )
    faithful_top_uncued, _, _ = get_aggregate_top_heads(
        faithful_pis, "uncued", False, save_dir, num_rollouts, top_k, drop_first
    )
    print(f"  Top CUED heads: {faithful_top_cued}")
    print(f"  Top UNCUED heads: {faithful_top_uncued}")

    # Unfaithful
    print(f"\n{'='*50}")
    print(f"UNFAITHFUL (PIs: {unfaithful_pis})")
    print(f"{'='*50}")
    unfaithful_top_cued, _, _ = get_aggregate_top_heads(
        unfaithful_pis, "cued", False, save_dir, num_rollouts, top_k, drop_first
    )
    unfaithful_top_uncued, _, _ = get_aggregate_top_heads(
        unfaithful_pis, "uncued", False, save_dir, num_rollouts, top_k, drop_first
    )
    print(f"  Top CUED heads: {unfaithful_top_cued}")
    print(f"  Top UNCUED heads: {unfaithful_top_uncued}")

    print("\n\nComputing aggregate top heads (REASONING ONLY)...")

    # Faithful reasoning
    faithful_top_cued_reasoning, _, _ = get_aggregate_top_heads(
        faithful_pis, "cued", True, save_dir, num_rollouts, top_k, drop_first
    )
    faithful_top_uncued_reasoning, _, _ = get_aggregate_top_heads(
        faithful_pis, "uncued", True, save_dir, num_rollouts, top_k, drop_first
    )

    # Unfaithful reasoning
    unfaithful_top_cued_reasoning, _, _ = get_aggregate_top_heads(
        unfaithful_pis, "cued", True, save_dir, num_rollouts, top_k, drop_first
    )
    unfaithful_top_uncued_reasoning, _, _ = get_aggregate_top_heads(
        unfaithful_pis, "uncued", True, save_dir, num_rollouts, top_k, drop_first
    )

    # Save faithful aggregate
    faithful_data = {
        "pis": faithful_pis,
        "num_rollouts_per_pi": num_rollouts,
        "total_rollouts": len(faithful_pis) * num_rollouts,
        "top_cued_heads": [(list(h), float(k)) for h, k in faithful_top_cued],
        "top_uncued_heads": [(list(h), float(k)) for h, k in faithful_top_uncued],
        "top_cued_reasoning_heads": [(list(h), float(k)) for h, k in faithful_top_cued_reasoning],
        "top_uncued_reasoning_heads": [(list(h), float(k)) for h, k in faithful_top_uncued_reasoning],
    }
    with open(os.path.join(aggregate_dir, "faithful", "aggregate_top_heads.json"), "w") as f:
        json.dump(faithful_data, f, indent=2)

    # Save unfaithful aggregate
    unfaithful_data = {
        "pis": unfaithful_pis,
        "num_rollouts_per_pi": num_rollouts,
        "total_rollouts": len(unfaithful_pis) * num_rollouts,
        "top_cued_heads": [(list(h), float(k)) for h, k in unfaithful_top_cued],
        "top_uncued_heads": [(list(h), float(k)) for h, k in unfaithful_top_uncued],
        "top_cued_reasoning_heads": [(list(h), float(k)) for h, k in unfaithful_top_cued_reasoning],
        "top_uncued_reasoning_heads": [(list(h), float(k)) for h, k in unfaithful_top_uncued_reasoning],
    }
    with open(os.path.join(aggregate_dir, "unfaithful", "aggregate_top_heads.json"), "w") as f:
        json.dump(unfaithful_data, f, indent=2)

    print(f"\n[OK] Saved aggregate results to {aggregate_dir}/")

    # Print comparison
    faithful_cued_set = {h for h, _ in faithful_top_cued}
    unfaithful_cued_set = {h for h, _ in unfaithful_top_cued}
    print(f"\n{'='*50}")
    print("COMPARISON - FULL (Cued condition)")
    print(f"{'='*50}")
    print(f"  Faithful-only heads:   {faithful_cued_set - unfaithful_cued_set}")
    print(f"  Unfaithful-only heads: {unfaithful_cued_set - faithful_cued_set}")
    print(f"  Shared heads:          {faithful_cued_set & unfaithful_cued_set}")


# =============================================================================
# Main Processing Function
# =============================================================================

def process_pi(model, tokenizer, pi, problem, category, save_dir, max_new_tokens,
               num_rollouts, top_k, drop_first, force=False):
    """Process a single problem ID."""
    save_path = get_save_path(pi, num_rollouts, top_k, drop_first, save_dir)

    # Check if already processed
    if os.path.exists(os.path.join(save_path, "config.json")) and not force:
        print(f"   [OK] Already processed, skipping! (use --force to re-run)")
        return {"pi": pi, "category": category, "status": "skipped"}

    # Generate rollouts
    print(f"\n   Generating {num_rollouts} UNCUED rollouts...")
    uncued_rollouts = run_rollouts(model, tokenizer, problem["question"], num_rollouts, max_new_tokens)

    print(f"\n   Generating {num_rollouts} CUED rollouts...")
    cued_rollouts = run_rollouts(model, tokenizer, problem["question_with_cue"], num_rollouts, max_new_tokens)

    # Collect vertical scores
    print(f"\n   Computing attention & vertical scores (UNCUED)...")
    uncued_h2v_full, uncued_h2v_reasoning, uncued_attentions = collect_vert_scores_for_rollouts(
        model, tokenizer, uncued_rollouts, cache_attention=True
    )

    print(f"\n   Computing attention & vertical scores (CUED)...")
    cued_h2v_full, cued_h2v_reasoning, cued_attentions = collect_vert_scores_for_rollouts(
        model, tokenizer, cued_rollouts, cache_attention=True
    )

    # Compute top heads - FULL
    uncued_head2kurt_full = compute_kurtosis_per_head(uncued_h2v_full)
    cued_head2kurt_full = compute_kurtosis_per_head(cued_h2v_full)

    top_uncued_full = select_top_heads(uncued_head2kurt_full, top_k=top_k,
                                        head2verts=uncued_h2v_full, min_max_vert=0.001)
    top_cued_full = select_top_heads(cued_head2kurt_full, top_k=top_k,
                                      head2verts=cued_h2v_full, min_max_vert=0.001)

    # Compute top heads - REASONING ONLY
    uncued_head2kurt_reasoning = compute_kurtosis_per_head(uncued_h2v_reasoning)
    cued_head2kurt_reasoning = compute_kurtosis_per_head(cued_h2v_reasoning)

    top_uncued_reasoning = select_top_heads(uncued_head2kurt_reasoning, top_k=top_k,
                                             head2verts=uncued_h2v_reasoning, min_max_vert=0.001)
    top_cued_reasoning = select_top_heads(cued_head2kurt_reasoning, top_k=top_k,
                                           head2verts=cued_h2v_reasoning, min_max_vert=0.001)

    # Select rollouts to save
    uncued_idx, uncued_rollout, uncued_attn = select_rollout_for_saving(
        uncued_rollouts, "uncued", uncued_attentions
    )
    cued_idx, cued_rollout, cued_attn = select_rollout_for_saving(
        cued_rollouts, "cued", cued_attentions
    )

    # Check for professor mentions
    cued_mentions = [i for i, ro in enumerate(cued_rollouts) if check_professor_mention(ro)]

    # Save results
    config = {
        "pi": pi,
        "category": category,
        "gt_answer": problem["gt_answer"],
        "cue_answer": problem["cue_answer"],
        "num_rollouts": num_rollouts,
        "top_k": top_k,
        "drop_first": drop_first,
        "proximity_ignore": PROXIMITY_IGNORE,
        "include_prompt": INCLUDE_PROMPT,
        "cued_rollout_idx": cued_idx,
        "uncued_rollout_idx": uncued_idx,
        "cued_professor_mentions": cued_mentions,
        "timestamp": datetime.now().isoformat(),
    }

    rollout_data = {
        "cued": cued_rollout,
        "uncued": uncued_rollout,
    }

    head2verts_data = {
        "cued_head2verts": cued_h2v_full,
        "uncued_head2verts": uncued_h2v_full,
        "cued_head2verts_reasoning": cued_h2v_reasoning,
        "uncued_head2verts_reasoning": uncued_h2v_reasoning,
    }

    top_heads_data = {
        "top_cued": top_cued_full,
        "top_uncued": top_uncued_full,
        "top_cued_reasoning": top_cued_reasoning,
        "top_uncued_reasoning": top_uncued_reasoning,
    }

    attention_data = {
        "cued": cued_attn,
        "uncued": uncued_attn,
    }

    save_analysis_results(save_path, config, rollout_data, head2verts_data, top_heads_data, attention_data)

    # Cleanup
    del uncued_rollouts, cued_rollouts
    del uncued_attentions, cued_attentions
    gc.collect()
    torch.cuda.empty_cache()

    return {"pi": pi, "category": category, "status": "completed"}


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run attention analysis for CoT faithfulness")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "gpqa"],
                        help="Dataset to process")
    parser.add_argument("--pis", type=int, nargs="+", default=None,
                        help="Specific problem IDs to process (default: all configured PIs)")
    parser.add_argument("--load-from-saved", action="store_true",
                        help="Skip PIs that already have saved data")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of all PIs (overwrite existing)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only compute aggregate results (skip rollout generation)")
    parser.add_argument("--num-rollouts", type=int, default=NUM_ROLLOUTS_PER_CONDITION,
                        help=f"Number of rollouts per condition (default: {NUM_ROLLOUTS_PER_CONDITION})")
    parser.add_argument("--top-k", type=int, default=TOP_K_RECEIVER_HEADS,
                        help=f"Number of top receiver heads (default: {TOP_K_RECEIVER_HEADS})")
    args = parser.parse_args()

    # Load config
    cfg = DATASET_CONFIG[args.dataset]
    problems_file = cfg["problems_file"]
    save_dir = cfg["save_dir"]
    max_new_tokens = cfg["max_new_tokens"]
    faithful_pis = cfg["faithful_pis"]
    unfaithful_pis = cfg["unfaithful_pis"]

    # Override with command line args
    if args.pis:
        all_pis = args.pis
        faithful_pis = [pi for pi in args.pis if pi in cfg["faithful_pis"]]
        unfaithful_pis = [pi for pi in args.pis if pi in cfg["unfaithful_pis"]]
    else:
        all_pis = faithful_pis + unfaithful_pis

    num_rollouts = args.num_rollouts
    top_k = args.top_k
    drop_first = DROP_FIRST

    print("=" * 60)
    print(f"Attention Analysis for {args.dataset.upper()}")
    print("=" * 60)
    print(f"Problems file: {problems_file}")
    print(f"Save directory: {save_dir}")
    print(f"PIs to process: {all_pis}")
    print(f"  Faithful: {faithful_pis}")
    print(f"  Unfaithful: {unfaithful_pis}")
    print(f"Rollouts per condition: {num_rollouts}")
    print(f"Top-K heads: {top_k}")
    print(f"Load from saved: {args.load_from_saved}")
    print(f"Force regeneration: {args.force}")

    # Load problems
    with open(problems_file, "r") as f:
        problems_data = json.load(f)

    pi_lookup = {}
    for cat in ["top_faithful", "top_unfaithful", "top_mixed"]:
        for p in problems_data.get(cat, []):
            pi_lookup[p["pi"]] = (p, cat)

    print(f"\nLoaded {len(pi_lookup)} problems from {problems_file}")

    if not args.aggregate_only:
        # Load model
        model, tokenizer = load_model()

        # Process each PI
        results_summary = []
        for pi_idx, pi in enumerate(all_pis):
            print(f"\n{'='*60}")
            print(f"Processing PI {pi} ({pi_idx + 1}/{len(all_pis)})")
            print(f"{'='*60}")

            if pi not in pi_lookup:
                print(f"   [!] PI {pi} not found in problems data, skipping!")
                continue

            problem, category = pi_lookup[pi]
            cat_label = "FAITHFUL" if pi in faithful_pis else "UNFAITHFUL"
            print(f"   Category: {cat_label} | GT={problem['gt_answer']} Cue={problem['cue_answer']}")

            # Check if should skip
            save_path = get_save_path(pi, num_rollouts, top_k, drop_first, save_dir)
            if args.load_from_saved and os.path.exists(os.path.join(save_path, "config.json")):
                print(f"   [OK] Already processed, skipping!")
                results_summary.append({"pi": pi, "category": cat_label, "status": "skipped"})
                continue

            result = process_pi(
                model, tokenizer, pi, problem, cat_label, save_dir, max_new_tokens,
                num_rollouts, top_k, drop_first, force=args.force
            )
            results_summary.append(result)

        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        for r in results_summary:
            status_icon = "[OK]" if r["status"] == "completed" else "[SKIP]"
            print(f"   {status_icon} PI {r['pi']} ({r['category']}): {r['status']}")

        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Compute aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE ANALYSIS")
    print(f"{'='*60}")
    save_aggregate_results(save_dir, faithful_pis, unfaithful_pis, num_rollouts, top_k, drop_first)

    print("\n[DONE] Analysis complete!")


if __name__ == "__main__":
    main()
