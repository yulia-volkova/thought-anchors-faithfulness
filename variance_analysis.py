#!/usr/bin/env python3
"""
Variance Analysis with Bootstrap Baseline for Thought Anchors Faithfulness.

This script computes in-group vs out-group variance using the same methodology
as the original thought-anchors paper:
1. Pool kurtosis values across all rollouts
2. Average kurtosis per head
3. Take top-k heads

For variance analysis:
- In-group: Bootstrap split rollouts within faithful/unfaithful, measure Jaccard
- Out-group: Jaccard between faithful aggregate and unfaithful aggregate

Usage:
    python variance_analysis.py --n-bootstrap 1000
    python variance_analysis.py --dataset mmlu
"""

import argparse
import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict
import random

# Dataset configurations (same as regenerate_aggregate.py)
DATASET_CONFIGS = {
    'mmlu': {
        'dir': 'final/mmlu',
        'faithful_pis': [91, 152, 188],
        'unfaithful_pis': [19, 151, 182, 191],
    },
    'gpqa': {
        'dir': 'final/gpqa',
        'faithful_pis': [162, 172, 129, 160, 21],
        'unfaithful_pis': [116, 101, 107, 100, 134],
    },
}

TOP_K_RECEIVER_HEADS = 5


def find_pi_folder(base_dir, pi):
    """Find the folder for a given PI (handles different naming conventions)."""
    for folder in os.listdir(base_dir):
        if folder.startswith(f"{pi}_") and os.path.isdir(os.path.join(base_dir, folder)):
            return os.path.join(base_dir, folder)
    return None


def load_head2verts(folder_path, condition="cued", reasoning_only=False):
    """Load head2verts for a single PI."""
    suffix = "_reasoning" if reasoning_only else ""
    path = os.path.join(folder_path, f"{condition}_head2verts{suffix}.json")

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    # Deserialize: "layer_head" -> (layer, head)
    h2v = {}
    for key, vs_list in data.items():
        l, h = map(int, key.split("_"))
        h2v[(l, h)] = [np.array(vs) for vs in vs_list]
    return h2v


def collect_all_kurtosis_values(base_dir, pi_list, condition="cued", reasoning_only=False):
    """
    Collect kurtosis values for each head from all rollouts.

    Returns:
        dict: {(layer, head): [list of (pi, rollout_idx, kurtosis_value)]}
    """
    head2kurt_records = defaultdict(list)

    for pi in pi_list:
        folder = find_pi_folder(base_dir, pi)
        if not folder:
            continue

        h2v = load_head2verts(folder, condition, reasoning_only=reasoning_only)
        if h2v is None:
            continue

        for (layer, head), vs_list in h2v.items():
            for rollout_idx, vs in enumerate(vs_list):
                if len(vs) > 3:
                    k = stats.kurtosis(vs, fisher=True, bias=True, nan_policy="omit")
                    if not np.isnan(k):
                        head2kurt_records[(layer, head)].append((pi, rollout_idx, k))

    return head2kurt_records


def get_top_heads_from_kurtosis_subset(head2kurt_records, subset_indices, top_k=TOP_K_RECEIVER_HEADS):
    """
    Compute top-k heads using only a subset of rollouts.

    Args:
        head2kurt_records: {(layer, head): [(pi, rollout_idx, kurt), ...]}
        subset_indices: set of (pi, rollout_idx) tuples to include
        top_k: number of top heads to return

    Returns:
        set of (layer, head) tuples
    """
    head2mean_kurt = {}

    for head, records in head2kurt_records.items():
        # Filter to subset
        subset_kurts = [k for (pi, ridx, k) in records if (pi, ridx) in subset_indices]
        if subset_kurts:
            head2mean_kurt[head] = np.mean(subset_kurts)

    # Get top-k
    items = [(h, v) for h, v in head2mean_kurt.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    return set(h for h, _ in items[:top_k])


def get_all_rollout_indices(head2kurt_records):
    """Get all unique (pi, rollout_idx) pairs from the records."""
    indices = set()
    for records in head2kurt_records.values():
        for (pi, ridx, _) in records:
            indices.add((pi, ridx))
    return list(indices)


def compute_jaccard(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def bootstrap_ingroup_jaccard(head2kurt_records, n_iterations=1000, top_k=TOP_K_RECEIVER_HEADS):
    """
    Bootstrap in-group consistency by splitting rollouts into two random halves.

    Returns:
        dict with mean, std, and percentiles
    """
    all_indices = get_all_rollout_indices(head2kurt_records)
    n = len(all_indices)

    if n < 4:
        return {'mean': None, 'std': None, 'n_rollouts': n, 'n_iterations': 0}

    jaccards = []
    half_size = n // 2

    for _ in range(n_iterations):
        # Random split
        shuffled = random.sample(all_indices, n)
        half1 = set(shuffled[:half_size])
        half2 = set(shuffled[half_size:])

        # Get top heads for each half
        heads1 = get_top_heads_from_kurtosis_subset(head2kurt_records, half1, top_k)
        heads2 = get_top_heads_from_kurtosis_subset(head2kurt_records, half2, top_k)

        j = compute_jaccard(heads1, heads2)
        jaccards.append(j)

    return {
        'mean': float(np.mean(jaccards)),
        'std': float(np.std(jaccards)),
        'percentile_5': float(np.percentile(jaccards, 5)),
        'percentile_95': float(np.percentile(jaccards, 95)),
        'n_rollouts': n,
        'n_iterations': n_iterations
    }


def compute_outgroup_jaccard(faithful_records, unfaithful_records, top_k=TOP_K_RECEIVER_HEADS):
    """
    Compute out-group Jaccard between aggregate faithful and unfaithful top heads.
    """
    # Get all indices for each group
    faithful_indices = set(get_all_rollout_indices(faithful_records))
    unfaithful_indices = set(get_all_rollout_indices(unfaithful_records))

    # Get top heads using all rollouts in each group
    faithful_heads = get_top_heads_from_kurtosis_subset(faithful_records, faithful_indices, top_k)
    unfaithful_heads = get_top_heads_from_kurtosis_subset(unfaithful_records, unfaithful_indices, top_k)

    return {
        'jaccard': compute_jaccard(faithful_heads, unfaithful_heads),
        'faithful_heads': [list(h) for h in faithful_heads],
        'unfaithful_heads': [list(h) for h in unfaithful_heads],
        'n_faithful_rollouts': len(faithful_indices),
        'n_unfaithful_rollouts': len(unfaithful_indices)
    }


def bootstrap_outgroup_baseline(faithful_records, unfaithful_records, n_iterations=1000, top_k=TOP_K_RECEIVER_HEADS):
    """
    Bootstrap baseline: randomly reassign rollouts to fake "faithful"/"unfaithful" groups.

    This tests whether the observed out-group divergence is different from random.
    """
    # Combine all records
    combined_records = defaultdict(list)
    for head, records in faithful_records.items():
        combined_records[head].extend(records)
    for head, records in unfaithful_records.items():
        combined_records[head].extend(records)

    all_indices = get_all_rollout_indices(combined_records)
    n = len(all_indices)

    # Use same group sizes as actual
    n_faithful = len(get_all_rollout_indices(faithful_records))
    n_unfaithful = len(get_all_rollout_indices(unfaithful_records))

    if n < 4:
        return {'mean': None, 'std': None, 'n_iterations': 0}

    jaccards = []

    for _ in range(n_iterations):
        # Random split into fake groups
        shuffled = random.sample(all_indices, n)
        fake_faithful = set(shuffled[:n_faithful])
        fake_unfaithful = set(shuffled[n_faithful:n_faithful + n_unfaithful])

        # Get top heads for each fake group
        heads1 = get_top_heads_from_kurtosis_subset(combined_records, fake_faithful, top_k)
        heads2 = get_top_heads_from_kurtosis_subset(combined_records, fake_unfaithful, top_k)

        j = compute_jaccard(heads1, heads2)
        jaccards.append(j)

    return {
        'mean': float(np.mean(jaccards)),
        'std': float(np.std(jaccards)),
        'percentile_5': float(np.percentile(jaccards, 5)),
        'percentile_95': float(np.percentile(jaccards, 95)),
        'n_iterations': n_iterations
    }


def compute_p_value(observed, bootstrap_mean, bootstrap_std):
    """Compute approximate p-value using normal approximation."""
    if bootstrap_std == 0:
        return None
    z = (observed - bootstrap_mean) / bootstrap_std
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(p_value)


def run_variance_analysis(dataset_name, n_bootstrap=1000):
    """Run complete variance analysis for a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    config = DATASET_CONFIGS[dataset_name]
    base_dir = config['dir']
    faithful_pis = config['faithful_pis']
    unfaithful_pis = config['unfaithful_pis']

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"Running variance analysis for {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Base dir: {base_dir}")
    print(f"Faithful PIs: {faithful_pis}")
    print(f"Unfaithful PIs: {unfaithful_pis}")

    results = {}

    for reasoning_only in [False, True]:
        mode = "reasoning_only" if reasoning_only else "full"
        print(f"\n--- Mode: {mode} ---")

        # Collect all kurtosis values
        print("  Collecting kurtosis values...")
        faithful_records = collect_all_kurtosis_values(
            base_dir, faithful_pis, "cued", reasoning_only
        )
        unfaithful_records = collect_all_kurtosis_values(
            base_dir, unfaithful_pis, "cued", reasoning_only
        )

        n_faithful = len(get_all_rollout_indices(faithful_records))
        n_unfaithful = len(get_all_rollout_indices(unfaithful_records))
        print(f"    Faithful: {n_faithful} rollouts")
        print(f"    Unfaithful: {n_unfaithful} rollouts")

        # In-group consistency (bootstrap)
        print(f"  Computing in-group consistency (bootstrap {n_bootstrap} iterations)...")
        faithful_ingroup = bootstrap_ingroup_jaccard(faithful_records, n_bootstrap)
        unfaithful_ingroup = bootstrap_ingroup_jaccard(unfaithful_records, n_bootstrap)

        if faithful_ingroup['mean'] is not None:
            print(f"    Faithful in-group: {faithful_ingroup['mean']:.3f} +/- {faithful_ingroup['std']:.3f}")
        else:
            print(f"    Faithful in-group: N/A (not enough rollouts)")

        if unfaithful_ingroup['mean'] is not None:
            print(f"    Unfaithful in-group: {unfaithful_ingroup['mean']:.3f} +/- {unfaithful_ingroup['std']:.3f}")
        else:
            print(f"    Unfaithful in-group: N/A (not enough rollouts)")

        # Out-group divergence
        print("  Computing out-group divergence...")
        outgroup = compute_outgroup_jaccard(faithful_records, unfaithful_records)
        print(f"    Out-group Jaccard: {outgroup['jaccard']:.3f}")
        print(f"    Faithful top-5: {outgroup['faithful_heads']}")
        print(f"    Unfaithful top-5: {outgroup['unfaithful_heads']}")

        # Bootstrap baseline for out-group
        print(f"  Computing random baseline (bootstrap {n_bootstrap} iterations)...")
        baseline = bootstrap_outgroup_baseline(faithful_records, unfaithful_records, n_bootstrap)

        if baseline['mean'] is not None:
            print(f"    Random baseline: {baseline['mean']:.3f} +/- {baseline['std']:.3f}")
            print(f"    95% CI: [{baseline['percentile_5']:.3f}, {baseline['percentile_95']:.3f}]")

            # Compute p-value
            p_value = compute_p_value(outgroup['jaccard'], baseline['mean'], baseline['std'])
            baseline['p_value'] = p_value
            if p_value is not None:
                sig = " ***" if p_value < 0.001 else " **" if p_value < 0.01 else " *" if p_value < 0.05 else ""
                print(f"    P-value: {p_value:.4f}{sig}")

        results[mode] = {
            'ingroup_jaccard': {
                'faithful': faithful_ingroup,
                'unfaithful': unfaithful_ingroup
            },
            'outgroup': outgroup,
            'baseline': baseline
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Variance analysis with bootstrap baseline")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to analyze (default: all)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--output", default="variance_analysis_results.json",
                        help="Output file path (default: variance_analysis_results.json)")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    all_results = {}

    if args.dataset:
        results = run_variance_analysis(args.dataset, args.n_bootstrap)
        if results:
            all_results[args.dataset] = results
    else:
        # Run for all datasets
        for dataset in DATASET_CONFIGS:
            results = run_variance_analysis(dataset, args.n_bootstrap)
            if results:
                all_results[dataset] = results

    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results to {args.output}")
    print(f"{'='*60}")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Done!")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        for mode in ['full', 'reasoning_only']:
            if mode in results:
                r = results[mode]
                print(f"\n  {mode}:")

                # In-group
                fi = r['ingroup_jaccard']['faithful']
                ui = r['ingroup_jaccard']['unfaithful']
                if fi['mean'] is not None:
                    print(f"    In-group (faithful):   {fi['mean']:.3f} +/- {fi['std']:.3f}")
                if ui['mean'] is not None:
                    print(f"    In-group (unfaithful): {ui['mean']:.3f} +/- {ui['std']:.3f}")

                # Out-group
                print(f"    Out-group Jaccard:     {r['outgroup']['jaccard']:.3f}")

                # Baseline
                if r['baseline']['mean'] is not None:
                    print(f"    Random baseline:       {r['baseline']['mean']:.3f} +/- {r['baseline']['std']:.3f}")
                    if r['baseline'].get('p_value') is not None:
                        print(f"    P-value:               {r['baseline']['p_value']:.4f}")


# ============================================================
# KURTOSIS MAGNITUDE ANALYSIS
# ============================================================
# Instead of comparing *which* heads are top receivers (identity),
# compare the *magnitude* of kurtosis between faithful and unfaithful

def compute_rollout_kurtosis_stats(head2kurt_records):
    """
    For each rollout, compute summary stats across all heads.

    Returns:
        list of dicts with {pi, rollout_idx, mean_kurt, max_kurt, median_kurt, std_kurt, n_heads}
    """
    # Group by (pi, rollout_idx)
    rollout2kurts = defaultdict(list)
    for head, records in head2kurt_records.items():
        for (pi, ridx, k) in records:
            rollout2kurts[(pi, ridx)].append(k)

    stats_list = []
    for (pi, ridx), kurts in rollout2kurts.items():
        if len(kurts) > 0:
            stats_list.append({
                'pi': pi,
                'rollout_idx': ridx,
                'mean_kurt': float(np.mean(kurts)),
                'max_kurt': float(np.max(kurts)),
                'median_kurt': float(np.median(kurts)),
                'std_kurt': float(np.std(kurts)),
                'n_heads': len(kurts)
            })
    return stats_list


def bootstrap_kurtosis_diff(faithful_stats, unfaithful_stats, metric='mean_kurt', n_iterations=1000):
    """
    Bootstrap CI for difference in kurtosis metric between faithful and unfaithful.

    Args:
        faithful_stats: list of dicts from compute_rollout_kurtosis_stats
        unfaithful_stats: list of dicts from compute_rollout_kurtosis_stats
        metric: which kurtosis metric to compare ('mean_kurt', 'max_kurt', 'median_kurt')
        n_iterations: number of bootstrap iterations

    Returns:
        dict with mean_diff, ci_95, p_value
    """
    f_vals = np.array([s[metric] for s in faithful_stats])
    u_vals = np.array([s[metric] for s in unfaithful_stats])

    observed_diff = float(np.mean(f_vals) - np.mean(u_vals))

    diffs = []
    for _ in range(n_iterations):
        f_sample = np.random.choice(f_vals, len(f_vals), replace=True)
        u_sample = np.random.choice(u_vals, len(u_vals), replace=True)
        diffs.append(np.mean(f_sample) - np.mean(u_sample))

    diffs = np.array(diffs)

    return {
        'observed_diff': observed_diff,
        'mean_diff': float(np.mean(diffs)),
        'std': float(np.std(diffs)),
        'ci_95': (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        # Two-tailed p-value
        'p_value': float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    }


def permutation_test_kurtosis(faithful_stats, unfaithful_stats, metric='mean_kurt', n_iterations=1000):
    """
    Permutation test for kurtosis difference between faithful and unfaithful.

    Args:
        faithful_stats: list of dicts from compute_rollout_kurtosis_stats
        unfaithful_stats: list of dicts from compute_rollout_kurtosis_stats
        metric: which kurtosis metric to compare
        n_iterations: number of permutations

    Returns:
        dict with observed_diff, null_mean, p_value
    """
    f_vals = [s[metric] for s in faithful_stats]
    u_vals = [s[metric] for s in unfaithful_stats]
    all_vals = f_vals + u_vals

    observed_diff = abs(np.mean(f_vals) - np.mean(u_vals))

    null_diffs = []
    for _ in range(n_iterations):
        shuffled = np.random.permutation(all_vals)
        fake_f = shuffled[:len(f_vals)]
        fake_u = shuffled[len(f_vals):]
        null_diffs.append(abs(np.mean(fake_f) - np.mean(fake_u)))

    null_diffs = np.array(null_diffs)
    p_value = np.mean(null_diffs >= observed_diff)

    return {
        'observed_diff': float(observed_diff),
        'null_mean': float(np.mean(null_diffs)),
        'null_std': float(np.std(null_diffs)),
        'p_value': float(p_value)
    }


def compute_effect_size(faithful_stats, unfaithful_stats, metric='mean_kurt'):
    """
    Compute Cohen's d effect size for the difference in kurtosis.
    """
    f_vals = np.array([s[metric] for s in faithful_stats])
    u_vals = np.array([s[metric] for s in unfaithful_stats])

    pooled_std = np.sqrt((np.var(f_vals) + np.var(u_vals)) / 2)

    if pooled_std == 0:
        return 0.0

    cohens_d = (np.mean(f_vals) - np.mean(u_vals)) / pooled_std
    return float(cohens_d)


def run_kurtosis_magnitude_analysis(dataset_name, n_bootstrap=1000):
    """
    Run kurtosis magnitude analysis for a dataset.

    Compares whether faithful and unfaithful rollouts have different
    kurtosis magnitudes (not just different head identities).
    """
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    config = DATASET_CONFIGS[dataset_name]
    base_dir = config['dir']
    faithful_pis = config['faithful_pis']
    unfaithful_pis = config['unfaithful_pis']

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"KURTOSIS MAGNITUDE ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*60}")

    results = {}

    for reasoning_only in [False, True]:
        mode = "reasoning_only" if reasoning_only else "full"
        print(f"\n--- Mode: {mode} ---")

        # Collect kurtosis values
        faithful_records = collect_all_kurtosis_values(
            base_dir, faithful_pis, "cued", reasoning_only
        )
        unfaithful_records = collect_all_kurtosis_values(
            base_dir, unfaithful_pis, "cued", reasoning_only
        )

        # Compute rollout-level stats
        faithful_stats = compute_rollout_kurtosis_stats(faithful_records)
        unfaithful_stats = compute_rollout_kurtosis_stats(unfaithful_records)

        print(f"  Faithful rollouts: {len(faithful_stats)}")
        print(f"  Unfaithful rollouts: {len(unfaithful_stats)}")

        if len(faithful_stats) < 3 or len(unfaithful_stats) < 3:
            print("  Not enough rollouts for analysis")
            continue

        mode_results = {}

        for metric in ['mean_kurt', 'max_kurt', 'median_kurt']:
            f_vals = [s[metric] for s in faithful_stats]
            u_vals = [s[metric] for s in unfaithful_stats]

            print(f"\n  {metric}:")
            print(f"    Faithful:   {np.mean(f_vals):.3f} ± {np.std(f_vals):.3f}")
            print(f"    Unfaithful: {np.mean(u_vals):.3f} ± {np.std(u_vals):.3f}")

            # Mann-Whitney U test
            from scipy.stats import mannwhitneyu
            stat, mw_pvalue = mannwhitneyu(f_vals, u_vals, alternative='two-sided')
            print(f"    Mann-Whitney U p-value: {mw_pvalue:.4f}")

            # Effect size
            d = compute_effect_size(faithful_stats, unfaithful_stats, metric)
            print(f"    Cohen's d: {d:.3f}")

            # Bootstrap CI
            boot_results = bootstrap_kurtosis_diff(faithful_stats, unfaithful_stats, metric, n_bootstrap)
            print(f"    Bootstrap 95% CI: [{boot_results['ci_95'][0]:.3f}, {boot_results['ci_95'][1]:.3f}]")

            # Permutation test
            perm_results = permutation_test_kurtosis(faithful_stats, unfaithful_stats, metric, n_bootstrap)
            print(f"    Permutation p-value: {perm_results['p_value']:.4f}")

            mode_results[metric] = {
                'faithful_mean': float(np.mean(f_vals)),
                'faithful_std': float(np.std(f_vals)),
                'unfaithful_mean': float(np.mean(u_vals)),
                'unfaithful_std': float(np.std(u_vals)),
                'mann_whitney_p': float(mw_pvalue),
                'cohens_d': d,
                'bootstrap': boot_results,
                'permutation': perm_results
            }

        results[mode] = {
            'n_faithful': len(faithful_stats),
            'n_unfaithful': len(unfaithful_stats),
            'metrics': mode_results
        }

    return results


if __name__ == "__main__":
    main()
