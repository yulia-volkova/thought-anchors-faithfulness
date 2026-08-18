#!/usr/bin/env python3
"""
Regenerate aggregate top heads data for all datasets.

This script computes aggregate receiver heads (by kurtosis) across all PIs
in each category (faithful/unfaithful) for both full attention and reasoning-only modes.

Kurtosis computation matches the original thought-anchors approach:
- Compute kurtosis per rollout (across sentences)
- Average kurtosis values across all rollouts from all PIs
See: github.com/interp-reasoning/thought-anchors/blob/main/whitebox-analyses/attention_analysis/receiver_head_funcs.py

Usage:
    python regenerate_aggregate.py --dataset mmlu
    python regenerate_aggregate.py --dataset gpqa
    python regenerate_aggregate.py  # regenerates all datasets
"""

import argparse
import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict

# Dataset configurations
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
    """Load head2verts for a single PI.

    Args:
        folder_path: Path to PI folder
        condition: "cued" or "uncued"
        reasoning_only: If True, load reasoning-only data (excludes prompt)
    """
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


def aggregate_kurtosis(base_dir, pi_list, condition="cued", reasoning_only=False):
    """
    Compute aggregate kurtosis from multiple PIs.
    Computes kurtosis PER ROLLOUT, then averages across all rollouts from all PIs.

    Matches original thought-anchors approach.
    """
    head2kurt_values = defaultdict(list)
    mode_str = "reasoning-only" if reasoning_only else "full"

    for pi in pi_list:
        folder = find_pi_folder(base_dir, pi)
        if not folder:
            print(f"   [!] PI {pi} folder not found, skipping")
            continue

        h2v = load_head2verts(folder, condition, reasoning_only=reasoning_only)
        if h2v is None:
            print(f"   [!] PI {pi} ({mode_str}) head2verts not found, skipping")
            continue

        for (layer, head), vs_list in h2v.items():
            for vs in vs_list:  # Each rollout independently
                if len(vs) > 3:  # Need enough points for kurtosis
                    k = stats.kurtosis(vs, fisher=True, bias=True, nan_policy="omit")
                    if not np.isnan(k):
                        head2kurt_values[(layer, head)].append(k)

    # Average kurtosis across ALL rollouts from ALL PIs
    head2kurt = {h: np.mean(ks) for h, ks in head2kurt_values.items() if len(ks) > 0}
    return head2kurt, head2kurt_values


def get_aggregate_top_heads(base_dir, pi_list, condition="cued", top_k=TOP_K_RECEIVER_HEADS, reasoning_only=False):
    """Get top heads for a category by aggregating across PIs."""
    head2kurt, _ = aggregate_kurtosis(base_dir, pi_list, condition, reasoning_only=reasoning_only)

    items = [(k, v) for k, v in head2kurt.items() if not np.isnan(v)]
    items.sort(key=lambda x: x[1], reverse=True)

    return items[:top_k]


def get_num_rollouts(base_dir, pi_list):
    """Get the number of rollouts per PI from config."""
    for pi in pi_list:
        folder = find_pi_folder(base_dir, pi)
        if folder:
            config_path = os.path.join(folder, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                return config.get("num_rollouts_per_condition", 5)
    return 5  # default


def regenerate_aggregate(dataset_name):
    """Regenerate aggregate data for a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    base_dir = config['dir']
    faithful_pis = config['faithful_pis']
    unfaithful_pis = config['unfaithful_pis']

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"Regenerating aggregate data for {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Base dir: {base_dir}")
    print(f"Faithful PIs: {faithful_pis}")
    print(f"Unfaithful PIs: {unfaithful_pis}")

    num_rollouts = get_num_rollouts(base_dir, faithful_pis + unfaithful_pis)
    print(f"Rollouts per PI: {num_rollouts}")

    # Create aggregate directory
    aggregate_dir = os.path.join(base_dir, "aggregate")
    os.makedirs(os.path.join(aggregate_dir, "faithful"), exist_ok=True)
    os.makedirs(os.path.join(aggregate_dir, "unfaithful"), exist_ok=True)

    # === FAITHFUL ===
    print(f"\n--- Faithful (PIs: {faithful_pis}) ---")

    # Full attention
    print("  Computing full attention heads...")
    faithful_top_cued = get_aggregate_top_heads(base_dir, faithful_pis, "cued", reasoning_only=False)
    faithful_top_uncued = get_aggregate_top_heads(base_dir, faithful_pis, "uncued", reasoning_only=False)
    print(f"    Top cued: {[h[0] for h in faithful_top_cued]}")
    print(f"    Top uncued: {[h[0] for h in faithful_top_uncued]}")

    # Reasoning only
    print("  Computing reasoning-only heads...")
    faithful_top_cued_reasoning = get_aggregate_top_heads(base_dir, faithful_pis, "cued", reasoning_only=True)
    faithful_top_uncued_reasoning = get_aggregate_top_heads(base_dir, faithful_pis, "uncued", reasoning_only=True)
    print(f"    Top cued (reasoning): {[h[0] for h in faithful_top_cued_reasoning]}")
    print(f"    Top uncued (reasoning): {[h[0] for h in faithful_top_uncued_reasoning]}")

    faithful_data = {
        "pis": faithful_pis,
        "num_rollouts_per_pi": num_rollouts,
        "total_rollouts": len(faithful_pis) * num_rollouts,
        # Full attention (with prompt)
        "top_cued_heads": [(list(h), float(k)) for h, k in faithful_top_cued],
        "top_uncued_heads": [(list(h), float(k)) for h, k in faithful_top_uncued],
        # Reasoning only (excludes prompt)
        "top_cued_reasoning_heads": [(list(h), float(k)) for h, k in faithful_top_cued_reasoning],
        "top_uncued_reasoning_heads": [(list(h), float(k)) for h, k in faithful_top_uncued_reasoning],
    }

    faithful_path = os.path.join(aggregate_dir, "faithful", "aggregate_top_heads.json")
    with open(faithful_path, "w") as f:
        json.dump(faithful_data, f, indent=2)
    print(f"  Saved: {faithful_path}")

    # === UNFAITHFUL ===
    print(f"\n--- Unfaithful (PIs: {unfaithful_pis}) ---")

    # Full attention
    print("  Computing full attention heads...")
    unfaithful_top_cued = get_aggregate_top_heads(base_dir, unfaithful_pis, "cued", reasoning_only=False)
    unfaithful_top_uncued = get_aggregate_top_heads(base_dir, unfaithful_pis, "uncued", reasoning_only=False)
    print(f"    Top cued: {[h[0] for h in unfaithful_top_cued]}")
    print(f"    Top uncued: {[h[0] for h in unfaithful_top_uncued]}")

    # Reasoning only
    print("  Computing reasoning-only heads...")
    unfaithful_top_cued_reasoning = get_aggregate_top_heads(base_dir, unfaithful_pis, "cued", reasoning_only=True)
    unfaithful_top_uncued_reasoning = get_aggregate_top_heads(base_dir, unfaithful_pis, "uncued", reasoning_only=True)
    print(f"    Top cued (reasoning): {[h[0] for h in unfaithful_top_cued_reasoning]}")
    print(f"    Top uncued (reasoning): {[h[0] for h in unfaithful_top_uncued_reasoning]}")

    unfaithful_data = {
        "pis": unfaithful_pis,
        "num_rollouts_per_pi": num_rollouts,
        "total_rollouts": len(unfaithful_pis) * num_rollouts,
        # Full attention (with prompt)
        "top_cued_heads": [(list(h), float(k)) for h, k in unfaithful_top_cued],
        "top_uncued_heads": [(list(h), float(k)) for h, k in unfaithful_top_uncued],
        # Reasoning only (excludes prompt)
        "top_cued_reasoning_heads": [(list(h), float(k)) for h, k in unfaithful_top_cued_reasoning],
        "top_uncued_reasoning_heads": [(list(h), float(k)) for h, k in unfaithful_top_uncued_reasoning],
    }

    unfaithful_path = os.path.join(aggregate_dir, "unfaithful", "aggregate_top_heads.json")
    with open(unfaithful_path, "w") as f:
        json.dump(unfaithful_data, f, indent=2)
    print(f"  Saved: {unfaithful_path}")

    print(f"\n[OK] Aggregate data regenerated for {dataset_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate aggregate top heads data")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to regenerate (default: all)")
    args = parser.parse_args()

    if args.dataset:
        regenerate_aggregate(args.dataset)
    else:
        # Regenerate all datasets
        for dataset in DATASET_CONFIGS:
            regenerate_aggregate(dataset)

    print("\n" + "="*60)
    print("Done! Now run: cd webapp && python preprocess_data.py")
    print("="*60)


if __name__ == "__main__":
    main()
