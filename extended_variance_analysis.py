#!/usr/bin/env python3
"""
Extended Variance Analysis for Thought Anchors Faithfulness.

This script computes additional statistical tests:
1. Permutation test for in-group consistency difference (faithful vs unfaithful)
2. Pooled dataset analysis (MMLU + GPQA combined)
3. Effect size calculations (Cohen's d)

Usage:
    python extended_variance_analysis.py
"""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict
import random
import math

# Import functions from the original variance_analysis.py
from variance_analysis import (
    DATASET_CONFIGS,
    TOP_K_RECEIVER_HEADS,
    find_pi_folder,
    load_head2verts,
    collect_all_kurtosis_values,
    get_all_rollout_indices,
    get_top_heads_from_kurtosis_subset,
    compute_jaccard,
    bootstrap_ingroup_jaccard,
    compute_outgroup_jaccard,
)


def permutation_test_ingroup_difference(
    faithful_records,
    unfaithful_records,
    n_permutations=10000,
    n_bootstrap_per_perm=100,
    top_k=TOP_K_RECEIVER_HEADS
):
    """
    Permutation test to compare in-group consistency between faithful and unfaithful.

    Tests null hypothesis: no difference in in-group Jaccard between conditions.

    Args:
        faithful_records: Kurtosis records for faithful rollouts
        unfaithful_records: Kurtosis records for unfaithful rollouts
        n_permutations: Number of permutation iterations
        n_bootstrap_per_perm: Bootstrap iterations per permutation (reduced for speed)
        top_k: Number of top heads to consider

    Returns:
        dict with observed difference, p-value, and null distribution stats
    """
    # Get observed in-group Jaccards
    faithful_ingroup = bootstrap_ingroup_jaccard(faithful_records, n_iterations=1000, top_k=top_k)
    unfaithful_ingroup = bootstrap_ingroup_jaccard(unfaithful_records, n_iterations=1000, top_k=top_k)

    if faithful_ingroup['mean'] is None or unfaithful_ingroup['mean'] is None:
        return {'error': 'Not enough data for bootstrap'}

    observed_diff = unfaithful_ingroup['mean'] - faithful_ingroup['mean']

    # Combine all records for permutation
    combined_records = defaultdict(list)
    for head, records in faithful_records.items():
        combined_records[head].extend(records)
    for head, records in unfaithful_records.items():
        combined_records[head].extend(records)

    all_indices = get_all_rollout_indices(combined_records)
    n_faithful = len(get_all_rollout_indices(faithful_records))
    n_unfaithful = len(get_all_rollout_indices(unfaithful_records))
    n_total = len(all_indices)

    print(f"    Permutation test: {n_faithful} faithful, {n_unfaithful} unfaithful, {n_total} total")
    print(f"    Observed difference: {observed_diff:.4f}")

    # Permutation test
    null_diffs = []

    for i in range(n_permutations):
        if (i + 1) % 1000 == 0:
            print(f"      Permutation {i + 1}/{n_permutations}")

        # Randomly shuffle labels
        shuffled = random.sample(all_indices, n_total)
        perm_faithful_indices = set(shuffled[:n_faithful])
        perm_unfaithful_indices = set(shuffled[n_faithful:n_faithful + n_unfaithful])

        # Create permuted records
        perm_faithful_records = defaultdict(list)
        perm_unfaithful_records = defaultdict(list)

        for head, records in combined_records.items():
            for (pi, ridx, k) in records:
                if (pi, ridx) in perm_faithful_indices:
                    perm_faithful_records[head].append((pi, ridx, k))
                elif (pi, ridx) in perm_unfaithful_indices:
                    perm_unfaithful_records[head].append((pi, ridx, k))

        # Compute in-group Jaccards for permuted groups
        perm_faithful_ingroup = bootstrap_ingroup_jaccard(
            perm_faithful_records, n_iterations=n_bootstrap_per_perm, top_k=top_k
        )
        perm_unfaithful_ingroup = bootstrap_ingroup_jaccard(
            perm_unfaithful_records, n_iterations=n_bootstrap_per_perm, top_k=top_k
        )

        if perm_faithful_ingroup['mean'] is not None and perm_unfaithful_ingroup['mean'] is not None:
            perm_diff = perm_unfaithful_ingroup['mean'] - perm_faithful_ingroup['mean']
            null_diffs.append(perm_diff)

    # Compute p-value (two-tailed)
    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    return {
        'observed_faithful_mean': faithful_ingroup['mean'],
        'observed_unfaithful_mean': unfaithful_ingroup['mean'],
        'observed_diff': float(observed_diff),
        'null_mean': float(np.mean(null_diffs)),
        'null_std': float(np.std(null_diffs)),
        'null_percentile_5': float(np.percentile(null_diffs, 5)),
        'null_percentile_95': float(np.percentile(null_diffs, 95)),
        'p_value': float(p_value),
        'n_permutations': len(null_diffs),
        'n_faithful': n_faithful,
        'n_unfaithful': n_unfaithful
    }


def compute_cohens_d(mean1, std1, mean2, std2):
    """Compute Cohen's d effect size."""
    pooled_std = math.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0:
        return None
    return (mean1 - mean2) / pooled_std


def run_pooled_analysis(n_bootstrap=1000):
    """
    Run variance analysis on pooled MMLU + GPQA data.
    """
    print("\n" + "=" * 60)
    print("POOLED ANALYSIS (MMLU + GPQA)")
    print("=" * 60)

    results = {}

    for reasoning_only in [False, True]:
        mode = "reasoning_only" if reasoning_only else "full"
        print(f"\n--- Mode: {mode} ---")

        # Collect from both datasets
        all_faithful_records = defaultdict(list)
        all_unfaithful_records = defaultdict(list)

        for dataset_name in ['mmlu', 'gpqa']:
            config = DATASET_CONFIGS[dataset_name]
            base_dir = config['dir']

            if not os.path.exists(base_dir):
                print(f"  Warning: {base_dir} not found, skipping")
                continue

            faithful_records = collect_all_kurtosis_values(
                base_dir, config['faithful_pis'], "cued", reasoning_only
            )
            unfaithful_records = collect_all_kurtosis_values(
                base_dir, config['unfaithful_pis'], "cued", reasoning_only
            )

            # Merge into pooled records
            for head, records in faithful_records.items():
                all_faithful_records[head].extend(records)
            for head, records in unfaithful_records.items():
                all_unfaithful_records[head].extend(records)

        n_faithful = len(get_all_rollout_indices(all_faithful_records))
        n_unfaithful = len(get_all_rollout_indices(all_unfaithful_records))
        print(f"  Pooled: {n_faithful} faithful, {n_unfaithful} unfaithful rollouts")

        # In-group consistency
        print(f"  Computing in-group consistency (bootstrap {n_bootstrap} iterations)...")
        faithful_ingroup = bootstrap_ingroup_jaccard(all_faithful_records, n_bootstrap)
        unfaithful_ingroup = bootstrap_ingroup_jaccard(all_unfaithful_records, n_bootstrap)

        if faithful_ingroup['mean'] is not None:
            print(f"    Faithful in-group: {faithful_ingroup['mean']:.3f} +/- {faithful_ingroup['std']:.3f}")
        if unfaithful_ingroup['mean'] is not None:
            print(f"    Unfaithful in-group: {unfaithful_ingroup['mean']:.3f} +/- {unfaithful_ingroup['std']:.3f}")

        # Out-group divergence
        print("  Computing out-group divergence...")
        outgroup = compute_outgroup_jaccard(all_faithful_records, all_unfaithful_records)
        print(f"    Out-group Jaccard: {outgroup['jaccard']:.3f}")
        print(f"    Faithful top-5: {outgroup['faithful_heads']}")
        print(f"    Unfaithful top-5: {outgroup['unfaithful_heads']}")

        # Effect size
        if faithful_ingroup['mean'] is not None and unfaithful_ingroup['mean'] is not None:
            cohens_d = compute_cohens_d(
                unfaithful_ingroup['mean'], unfaithful_ingroup['std'],
                faithful_ingroup['mean'], faithful_ingroup['std']
            )
            print(f"    Cohen's d: {cohens_d:.3f}")
        else:
            cohens_d = None

        results[mode] = {
            'ingroup_jaccard': {
                'faithful': faithful_ingroup,
                'unfaithful': unfaithful_ingroup
            },
            'outgroup': outgroup,
            'n_faithful': n_faithful,
            'n_unfaithful': n_unfaithful,
            'cohens_d': cohens_d
        }

    return results


def run_ingroup_permutation_tests(n_permutations=1000, n_bootstrap_per_perm=50):
    """
    Run permutation tests for in-group consistency difference.
    """
    print("\n" + "=" * 60)
    print("PERMUTATION TESTS FOR IN-GROUP CONSISTENCY DIFFERENCE")
    print("=" * 60)

    results = {}

    for dataset_name in ['mmlu', 'gpqa']:
        config = DATASET_CONFIGS[dataset_name]
        base_dir = config['dir']

        if not os.path.exists(base_dir):
            print(f"  Warning: {base_dir} not found, skipping")
            continue

        print(f"\n{dataset_name.upper()}:")

        for reasoning_only in [False, True]:
            mode = "reasoning_only" if reasoning_only else "full"
            print(f"\n  Mode: {mode}")

            faithful_records = collect_all_kurtosis_values(
                base_dir, config['faithful_pis'], "cued", reasoning_only
            )
            unfaithful_records = collect_all_kurtosis_values(
                base_dir, config['unfaithful_pis'], "cued", reasoning_only
            )

            perm_result = permutation_test_ingroup_difference(
                faithful_records,
                unfaithful_records,
                n_permutations=n_permutations,
                n_bootstrap_per_perm=n_bootstrap_per_perm
            )

            if 'error' in perm_result:
                print(f"    Error: {perm_result['error']}")
            else:
                sig = " ***" if perm_result['p_value'] < 0.001 else " **" if perm_result['p_value'] < 0.01 else " *" if perm_result['p_value'] < 0.05 else ""
                print(f"    Observed diff: {perm_result['observed_diff']:.4f}")
                print(f"    Null distribution: {perm_result['null_mean']:.4f} +/- {perm_result['null_std']:.4f}")
                print(f"    P-value: {perm_result['p_value']:.4f}{sig}")

                # Effect size
                cohens_d = compute_cohens_d(
                    perm_result['observed_unfaithful_mean'], 0.06,  # approx std from original
                    perm_result['observed_faithful_mean'], 0.10
                )
                if cohens_d:
                    print(f"    Cohen's d: {cohens_d:.3f}")

            if dataset_name not in results:
                results[dataset_name] = {}
            results[dataset_name][mode] = perm_result

    return results


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Change to the correct directory
    os.chdir('/Users/yuliav/PycharmProjects/thought-anchors-faithfulness')

    all_results = {}

    # 1. Run permutation tests for in-group consistency
    print("\nRunning permutation tests (this may take a few minutes)...")
    perm_results = run_ingroup_permutation_tests(
        n_permutations=1000,  # Reduced for speed; increase for final analysis
        n_bootstrap_per_perm=50
    )
    all_results['permutation_tests'] = perm_results

    # 2. Run pooled analysis
    pooled_results = run_pooled_analysis(n_bootstrap=1000)
    all_results['pooled'] = pooled_results

    # 3. Compute effect sizes for original data
    print("\n" + "=" * 60)
    print("EFFECT SIZE SUMMARY")
    print("=" * 60)

    # Load original results
    with open('variance_analysis_results.json', 'r') as f:
        original_results = json.load(f)

    effect_sizes = {}
    for dataset in ['mmlu', 'gpqa']:
        effect_sizes[dataset] = {}
        for mode in ['full', 'reasoning_only']:
            if dataset in original_results and mode in original_results[dataset]:
                r = original_results[dataset][mode]
                fi = r['ingroup_jaccard']['faithful']
                ui = r['ingroup_jaccard']['unfaithful']

                if fi['mean'] is not None and ui['mean'] is not None:
                    d = compute_cohens_d(ui['mean'], ui['std'], fi['mean'], fi['std'])
                    effect_sizes[dataset][mode] = {
                        'cohens_d': d,
                        'faithful_mean': fi['mean'],
                        'unfaithful_mean': ui['mean']
                    }
                    print(f"{dataset.upper()} {mode}: d = {d:.3f} (unfaithful - faithful)")

    all_results['effect_sizes'] = effect_sizes

    # Save results
    output_file = 'extended_variance_analysis_results.json'
    print(f"\n{'='*60}")
    print(f"Saving results to {output_file}")
    print(f"{'='*60}")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF NEW FINDINGS")
    print("=" * 60)

    print("\n1. PERMUTATION TEST P-VALUES (In-group consistency difference):")
    for dataset, modes in perm_results.items():
        for mode, result in modes.items():
            if 'p_value' in result:
                sig = "SIGNIFICANT" if result['p_value'] < 0.05 else "not significant"
                print(f"   {dataset.upper()} {mode}: p = {result['p_value']:.4f} ({sig})")

    print("\n2. POOLED ANALYSIS (MMLU + GPQA combined):")
    for mode, result in pooled_results.items():
        fi = result['ingroup_jaccard']['faithful']
        ui = result['ingroup_jaccard']['unfaithful']
        if fi['mean'] is not None and ui['mean'] is not None:
            print(f"   {mode}: faithful={fi['mean']:.3f}, unfaithful={ui['mean']:.3f}, d={result['cohens_d']:.3f}")

    print("\n3. EFFECT SIZES (Cohen's d):")
    for dataset, modes in effect_sizes.items():
        for mode, result in modes.items():
            if 'cohens_d' in result:
                size = "large" if abs(result['cohens_d']) > 0.8 else "medium" if abs(result['cohens_d']) > 0.5 else "small"
                print(f"   {dataset.upper()} {mode}: d = {result['cohens_d']:.3f} ({size})")

    print("\nDone!")


if __name__ == "__main__":
    main()
