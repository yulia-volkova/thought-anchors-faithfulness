#!/usr/bin/env python3
"""
Analyze attention patterns and faithfulness results.

This script analyzes the relationship between:
- Cue attention (whether the cue sentence is a top attention target)
- Faithfulness (whether the model mentions the cue in reasoning)
- Accuracy (whether the model gets the correct answer)

Results are broken down by dataset (MMLU vs GPQA).
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats

# Configuration
BASE_DIR = Path(__file__).parent
FINAL_DIR = BASE_DIR / 'final'
CUE_PATTERNS = ['professor', 'stanford', 'iq of 130', 'iq 130']

# PI filters (matching webapp)
ALLOWED_GPQA_PIS = [21, 100, 107, 116, 129, 160, 162, 172]


@dataclass
class RolloutAnalysis:
    """Analysis results for a single rollout."""
    pi: int
    dataset: str
    condition: str  # 'cued', 'uncued', 'faithful', 'unfaithful'
    answer: Optional[str]
    gt_answer: str
    cue_answer: str
    is_correct: bool
    follows_cue: bool
    mentions_cue: bool  # Whether cue is mentioned in reasoning text
    prompt_len: int
    num_sentences: int
    # Attention analysis
    cue_in_top1: bool = False
    cue_in_top3: bool = False
    cue_in_top5: bool = False
    prompt_in_top1: bool = False
    prompt_in_top3: bool = False
    prompt_in_top5: bool = False
    cue_sentence_rank: Optional[int] = None
    cue_sentence_attn: Optional[float] = None
    top_sources: list = field(default_factory=list)


def has_cue(text: str) -> bool:
    """Check if text contains cue patterns."""
    text_lower = text.lower()
    return any(pat in text_lower for pat in CUE_PATTERNS)


def rank_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Apply rank normalization to attention matrix (matching original thought-anchors)."""
    matrix = matrix.copy()
    # Set upper triangle to NaN (only keep lower triangle)
    triu_indices = np.triu_indices_from(matrix, k=1)
    matrix[triu_indices] = np.nan

    # Rank normalize each row
    per_row = np.sum(~np.isnan(matrix), axis=1)
    per_row = np.where(per_row == 0, 1, per_row)
    matrix = stats.rankdata(matrix, axis=1, nan_policy='omit') / per_row[:, None]

    # Set upper triangle to 0
    matrix[triu_indices] = 0
    return matrix


def compute_column_means(matrix: np.ndarray) -> np.ndarray:
    """Compute column means (attention received by each source from later sentences)."""
    n = matrix.shape[0]
    col_means = np.zeros(n)

    for j in range(n):
        # Only look at lower triangle (later attending to earlier)
        col_vals = matrix[j+1:, j]
        non_zero = col_vals[col_vals > 0]
        if len(non_zero) > 0:
            col_means[j] = np.mean(non_zero)

    return col_means


def load_attention_and_analyze(attn_path: Path, rollout_path: Path, problem_info: dict) -> Optional[RolloutAnalysis]:
    """Load attention data and analyze a single rollout."""
    if not attn_path.exists() or not rollout_path.exists():
        return None

    # Load rollout metadata
    with open(rollout_path) as f:
        rollout = json.load(f)

    sentences = rollout.get('sentences', [])
    prompt_len = rollout.get('prompt_len', 0)

    if not sentences:
        return None

    # Load and average attention matrices across all heads
    attn_data = np.load(attn_path, allow_pickle=True)
    matrices = []
    for key in attn_data.keys():
        mat = attn_data[key].astype(np.float32)
        matrices.append(mat)

    if not matrices:
        return None

    # Average across heads
    avg_matrix = np.mean(matrices, axis=0)

    # Rank normalize
    avg_matrix = rank_normalize_matrix(avg_matrix)

    # Compute column means
    col_means = compute_column_means(avg_matrix)

    # Sort to find top sources
    sorted_indices = np.argsort(col_means)[::-1]

    # Analyze top sources
    top_sources = []
    cue_rank = None
    cue_attn = None

    for rank, idx in enumerate(sorted_indices[:10]):  # Top 10
        idx = int(idx)  # Convert from numpy int
        if idx < len(sentences):
            sent = sentences[idx]
            is_cue = bool(has_cue(sent))
            is_prompt = bool(idx < prompt_len)

            if is_cue and cue_rank is None:
                cue_rank = rank + 1  # 1-indexed
                cue_attn = float(col_means[idx])

            top_sources.append({
                'rank': rank + 1,
                'idx': idx,
                'sentence': sent[:100],
                'attn': float(col_means[idx]),
                'is_cue': is_cue,
                'is_prompt': is_prompt
            })

    # Determine cue/prompt in top N
    top1_sources = top_sources[:1]
    top3_sources = top_sources[:3]
    top5_sources = top_sources[:5]

    cue_in_top1 = bool(any(s['is_cue'] for s in top1_sources))
    cue_in_top3 = bool(any(s['is_cue'] for s in top3_sources))
    cue_in_top5 = bool(any(s['is_cue'] for s in top5_sources))
    prompt_in_top1 = bool(any(s['is_prompt'] for s in top1_sources))
    prompt_in_top3 = bool(any(s['is_prompt'] for s in top3_sources))
    prompt_in_top5 = bool(any(s['is_prompt'] for s in top5_sources))

    # Extract answer from rollout text (simple heuristic)
    text = rollout.get('text', '')
    answer = None
    for line in text.split('\n')[::-1]:
        if 'the best answer is' in line.lower():
            for letter in ['A', 'B', 'C', 'D']:
                if f'({letter})' in line or f': {letter}' in line or f'is {letter}' in line:
                    answer = letter
                    break
            if answer:
                break

    gt_answer = problem_info.get('gt_answer', '')
    cue_answer = problem_info.get('cue_answer', '')

    return RolloutAnalysis(
        pi=problem_info.get('pi', 0),
        dataset=problem_info.get('dataset', ''),
        condition=problem_info.get('condition', ''),
        answer=answer,
        gt_answer=gt_answer,
        cue_answer=cue_answer,
        is_correct=(answer == gt_answer) if answer else False,
        follows_cue=(answer == cue_answer) if answer else False,
        mentions_cue=has_cue(text),
        prompt_len=prompt_len,
        num_sentences=len(sentences),
        cue_in_top1=cue_in_top1,
        cue_in_top3=cue_in_top3,
        cue_in_top5=cue_in_top5,
        prompt_in_top1=prompt_in_top1,
        prompt_in_top3=prompt_in_top3,
        prompt_in_top5=prompt_in_top5,
        cue_sentence_rank=cue_rank,
        cue_sentence_attn=cue_attn,
        top_sources=top_sources
    )


def load_all_rollouts() -> list[RolloutAnalysis]:
    """Load and analyze all rollouts from the final directory."""
    rollouts = []

    for dataset in ['mmlu', 'gpqa']:
        dataset_dir = FINAL_DIR / dataset
        if not dataset_dir.exists():
            continue

        for pi_dir in dataset_dir.iterdir():
            if not pi_dir.is_dir():
                continue

            # Parse PI from directory name (format: {pi}_5_5_1)
            try:
                pi = int(pi_dir.name.split('_')[0])
            except ValueError:
                continue

            # Filter GPQA PIs
            if dataset == 'gpqa' and pi not in ALLOWED_GPQA_PIS:
                continue

            # Load problem info
            problem_info_path = pi_dir / 'problem_info.json'
            if problem_info_path.exists():
                with open(problem_info_path) as f:
                    problem_info = json.load(f)
            else:
                problem_info = {}

            problem_info['pi'] = pi
            problem_info['dataset'] = dataset

            # Process cued rollout
            cued_dir = pi_dir / 'cued'
            if cued_dir.exists():
                problem_info['condition'] = 'cued'
                analysis = load_attention_and_analyze(
                    cued_dir / 'attention.npz',
                    cued_dir / 'rollout.json',
                    problem_info
                )
                if analysis:
                    rollouts.append(analysis)

            # Process uncued rollout
            uncued_dir = pi_dir / 'uncued'
            if uncued_dir.exists():
                problem_info['condition'] = 'uncued'
                analysis = load_attention_and_analyze(
                    uncued_dir / 'attention.npz',
                    uncued_dir / 'rollout.json',
                    problem_info
                )
                if analysis:
                    rollouts.append(analysis)

            # Process faithful/unfaithful rollouts
            fvu_dir = pi_dir / 'faithful_vs_unfaithful'
            if fvu_dir.exists():
                for condition in ['faithful', 'unfaithful']:
                    condition_dir = fvu_dir / condition
                    if condition_dir.exists():
                        problem_info['condition'] = condition
                        analysis = load_attention_and_analyze(
                            condition_dir / 'attention.npz',
                            condition_dir / 'rollout.json',
                            problem_info
                        )
                        if analysis:
                            rollouts.append(analysis)

    return rollouts


def print_analysis(rollouts: list[RolloutAnalysis]):
    """Print comprehensive analysis of rollouts."""
    print("=" * 80)
    print("ATTENTION & FAITHFULNESS ANALYSIS")
    print("=" * 80)

    # Group by dataset
    by_dataset = defaultdict(list)
    for r in rollouts:
        by_dataset[r.dataset].append(r)

    for dataset in ['mmlu', 'gpqa']:
        dataset_rollouts = by_dataset[dataset]
        if not dataset_rollouts:
            continue

        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*80}")

        # Group by condition
        by_condition = defaultdict(list)
        for r in dataset_rollouts:
            by_condition[r.condition].append(r)

        print(f"\nTotal rollouts: {len(dataset_rollouts)}")
        for cond in ['cued', 'uncued', 'faithful', 'unfaithful']:
            if cond in by_condition:
                print(f"  {cond}: {len(by_condition[cond])}")

        # Analyze faithful vs unfaithful
        for condition in ['faithful', 'unfaithful']:
            cond_rollouts = by_condition.get(condition, [])
            if not cond_rollouts:
                continue

            n = len(cond_rollouts)
            print(f"\n--- {condition.upper()} ROLLOUTS (n={n}) ---")

            # Cue in top sources
            cue_top1 = sum(1 for r in cond_rollouts if r.cue_in_top1)
            cue_top3 = sum(1 for r in cond_rollouts if r.cue_in_top3)
            cue_top5 = sum(1 for r in cond_rollouts if r.cue_in_top5)
            print(f"\nCue sentence in top attention sources:")
            print(f"  Top 1: {cue_top1}/{n} ({cue_top1/n*100:.1f}%)")
            print(f"  Top 3: {cue_top3}/{n} ({cue_top3/n*100:.1f}%)")
            print(f"  Top 5: {cue_top5}/{n} ({cue_top5/n*100:.1f}%)")

            # Prompt in top sources
            prompt_top1 = sum(1 for r in cond_rollouts if r.prompt_in_top1)
            prompt_top3 = sum(1 for r in cond_rollouts if r.prompt_in_top3)
            prompt_top5 = sum(1 for r in cond_rollouts if r.prompt_in_top5)
            print(f"\nPrompt sentence in top attention sources:")
            print(f"  Top 1: {prompt_top1}/{n} ({prompt_top1/n*100:.1f}%)")
            print(f"  Top 3: {prompt_top3}/{n} ({prompt_top3/n*100:.1f}%)")
            print(f"  Top 5: {prompt_top5}/{n} ({prompt_top5/n*100:.1f}%)")

            # Accuracy
            correct = sum(1 for r in cond_rollouts if r.is_correct)
            print(f"\nAccuracy: {correct}/{n} ({correct/n*100:.1f}%)")

            # Cue following
            follows = sum(1 for r in cond_rollouts if r.follows_cue)
            print(f"Cue following: {follows}/{n} ({follows/n*100:.1f}%)")

            # Cue rank statistics
            cue_ranks = [r.cue_sentence_rank for r in cond_rollouts if r.cue_sentence_rank]
            if cue_ranks:
                print(f"\nCue sentence rank (when in top 10):")
                print(f"  Mean: {np.mean(cue_ranks):.1f}")
                print(f"  Median: {np.median(cue_ranks):.1f}")
                print(f"  Range: {min(cue_ranks)} - {max(cue_ranks)}")

        # Cross-analysis: Cue attention vs accuracy
        print(f"\n--- CUE ATTENTION vs ACCURACY ---")
        fvu_rollouts = by_condition.get('faithful', []) + by_condition.get('unfaithful', [])

        if fvu_rollouts:
            with_cue = [r for r in fvu_rollouts if r.cue_in_top5]
            without_cue = [r for r in fvu_rollouts if not r.cue_in_top5]

            if with_cue:
                acc_with = sum(1 for r in with_cue if r.is_correct) / len(with_cue) * 100
                print(f"Accuracy when cue in top 5: {acc_with:.1f}% (n={len(with_cue)})")
            if without_cue:
                acc_without = sum(1 for r in without_cue if r.is_correct) / len(without_cue) * 100
                print(f"Accuracy when cue NOT in top 5: {acc_without:.1f}% (n={len(without_cue)})")

        # Cross-analysis: Cue attention vs faithfulness
        print(f"\n--- CUE ATTENTION vs FAITHFULNESS ---")
        if fvu_rollouts:
            with_cue = [r for r in fvu_rollouts if r.cue_in_top5]
            without_cue = [r for r in fvu_rollouts if not r.cue_in_top5]

            if with_cue:
                faith_with = sum(1 for r in with_cue if r.condition == 'faithful') / len(with_cue) * 100
                print(f"Faithful when cue in top 5: {faith_with:.1f}% (n={len(with_cue)})")
            if without_cue:
                faith_without = sum(1 for r in without_cue if r.condition == 'faithful') / len(without_cue) * 100
                print(f"Faithful when cue NOT in top 5: {faith_without:.1f}% (n={len(without_cue)})")

    # Key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    print("\n1. UNFAITHFUL ROLLOUTS WITH HIGH CUE ATTENTION")
    print("   (Model attends to cue but doesn't mention it in reasoning)")
    for dataset in ['mmlu', 'gpqa']:
        unfaithful = [r for r in by_dataset[dataset] if r.condition == 'unfaithful']
        if unfaithful:
            cue_attn = sum(1 for r in unfaithful if r.cue_in_top5)
            print(f"   {dataset.upper()}: {cue_attn}/{len(unfaithful)} ({cue_attn/len(unfaithful)*100:.1f}%)")

    print("\n2. TOP SOURCE IS PROMPT SENTENCE")
    for dataset in ['mmlu', 'gpqa']:
        fvu = [r for r in by_dataset[dataset] if r.condition in ['faithful', 'unfaithful']]
        if fvu:
            prompt_top1 = sum(1 for r in fvu if r.prompt_in_top1)
            print(f"   {dataset.upper()}: {prompt_top1}/{len(fvu)} ({prompt_top1/len(fvu)*100:.1f}%)")


def export_for_notebook(rollouts: list[RolloutAnalysis], output_path: Path):
    """Export rollout data as JSON for notebook analysis."""
    data = []
    for r in rollouts:
        data.append({
            'pi': r.pi,
            'dataset': r.dataset,
            'condition': r.condition,
            'answer': r.answer,
            'gt_answer': r.gt_answer,
            'cue_answer': r.cue_answer,
            'is_correct': bool(r.is_correct),
            'follows_cue': bool(r.follows_cue),
            'mentions_cue': bool(r.mentions_cue),
            'prompt_len': r.prompt_len,
            'num_sentences': r.num_sentences,
            'cue_in_top1': bool(r.cue_in_top1),
            'cue_in_top3': bool(r.cue_in_top3),
            'cue_in_top5': bool(r.cue_in_top5),
            'prompt_in_top1': bool(r.prompt_in_top1),
            'prompt_in_top3': bool(r.prompt_in_top3),
            'prompt_in_top5': bool(r.prompt_in_top5),
            'cue_sentence_rank': int(r.cue_sentence_rank) if r.cue_sentence_rank else None,
            'cue_sentence_attn': float(r.cue_sentence_attn) if r.cue_sentence_attn else None,
            'top_sources': r.top_sources
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nExported {len(data)} rollouts to {output_path}")


def main():
    print("Loading rollouts...")
    rollouts = load_all_rollouts()
    print(f"Loaded {len(rollouts)} rollouts")

    print_analysis(rollouts)

    # Export for notebook
    export_for_notebook(rollouts, BASE_DIR / 'analysis_results.json')


if __name__ == '__main__':
    main()
