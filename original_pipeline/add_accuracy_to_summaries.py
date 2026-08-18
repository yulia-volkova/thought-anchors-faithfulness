"""
Script to compute accuracy metrics from long datasets and add them to summary tables.

Accuracy is computed as the proportion of rollouts per pi where answer == gt_answer.
This is computed separately for cue and base conditions.
"""

import argparse
import pandas as pd
from datasets import Dataset

from hf_utils import load_hf_as_df


def compute_accuracy_per_pi(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy per pi from long dataset.
    
    Accuracy = proportion of rollouts where answer == gt_answer (excluding null answers).
    
    Returns:
        DataFrame with columns: pi, accuracy, n_valid_responses
    """
    # Filter out null answers
    df_valid = df_long[df_long["answer"].notna()].copy()
    
    # Compute accuracy per pi
    accuracy_stats = []
    for pi in df_valid["pi"].unique():
        pi_data = df_valid[df_valid["pi"] == pi]
        n_valid = len(pi_data)
        n_correct = (pi_data["answer"] == pi_data["gt_answer"]).sum()
        accuracy = n_correct / n_valid if n_valid > 0 else 0.0
        
        accuracy_stats.append({
            "pi": pi,
            "accuracy": accuracy,
            "n_valid_responses": n_valid,
        })
    
    return pd.DataFrame(accuracy_stats)


def add_accuracy_to_summaries(
    cue_long_dataset: str = "yulia-volkova/mmlu-chua-cue-long",
    base_long_dataset: str = "yulia-volkova/mmlu-chua-base-long",
    cue_summary_dataset: str = "yulia-volkova/mmlu-chua-cue-summary",
    base_summary_dataset: str = "yulia-volkova/mmlu-chua-base-summary",
    push_to_hub: bool = False,
    dry_run: bool = False,
):
    """
    Load long datasets, compute accuracy per pi, and add to summary datasets.
    """
    print("="*60)
    print("Adding Accuracy Metrics to Summary Tables")
    print("="*60)
    
    # Load long datasets
    print(f"\n1. Loading long datasets...")
    print(f"   Loading {cue_long_dataset}...")
    df_cue_long = load_hf_as_df(cue_long_dataset)
    print(f"   Loaded {len(df_cue_long)} rollouts")
    
    print(f"   Loading {base_long_dataset}...")
    df_base_long = load_hf_as_df(base_long_dataset)
    print(f"   Loaded {len(df_base_long)} rollouts")
    
    # Compute accuracy per pi
    print(f"\n2. Computing accuracy per pi...")
    print(f"   Computing accuracy for cue condition...")
    cue_accuracy = compute_accuracy_per_pi(df_cue_long)
    print(f"   Computed accuracy for {len(cue_accuracy)} problems")
    
    print(f"   Computing accuracy for base condition...")
    base_accuracy = compute_accuracy_per_pi(df_base_long)
    print(f"   Computed accuracy for {len(base_accuracy)} problems")
    
    # Load summary datasets
    print(f"\n3. Loading summary datasets...")
    print(f"   Loading {cue_summary_dataset}...")
    df_cue_summary = load_hf_as_df(cue_summary_dataset)
    print(f"   Loaded {len(df_cue_summary)} problems")
    
    print(f"   Loading {base_summary_dataset}...")
    df_base_summary = load_hf_as_df(base_summary_dataset)
    print(f"   Loaded {len(df_base_summary)} problems")
    
    # Merge accuracy into summaries
    print(f"\n4. Merging accuracy into summaries...")
    
    # Check if accuracy column already exists
    if "accuracy" in df_cue_summary.columns:
        print("   Warning: 'accuracy' column already exists in cue summary. Overwriting...")
        df_cue_summary = df_cue_summary.drop(columns=["accuracy"])
    
    if "accuracy" in df_base_summary.columns:
        print("   Warning: 'accuracy' column already exists in base summary. Overwriting...")
        df_base_summary = df_base_summary.drop(columns=["accuracy"])
    
    # Merge
    df_cue_summary = pd.merge(
        df_cue_summary,
        cue_accuracy[["pi", "accuracy"]],
        on="pi",
        how="left",
    )
    
    df_base_summary = pd.merge(
        df_base_summary,
        base_accuracy[["pi", "accuracy"]],
        on="pi",
        how="left",
    )
    
    # Check for missing values
    cue_missing = df_cue_summary["accuracy"].isna().sum()
    base_missing = df_base_summary["accuracy"].isna().sum()
    
    if cue_missing > 0:
        print(f"   Warning: {cue_missing} problems in cue summary have no accuracy data")
    if base_missing > 0:
        print(f"   Warning: {base_missing} problems in base summary have no accuracy data")
    
    # Print summary statistics
    print(f"\n5. Accuracy Statistics:")
    print(f"   Cue condition:")
    print(f"     Mean accuracy: {df_cue_summary['accuracy'].mean():.3f}")
    print(f"     Median accuracy: {df_cue_summary['accuracy'].median():.3f}")
    print(f"     Min accuracy: {df_cue_summary['accuracy'].min():.3f}")
    print(f"     Max accuracy: {df_cue_summary['accuracy'].max():.3f}")
    
    print(f"   Base condition:")
    print(f"     Mean accuracy: {df_base_summary['accuracy'].mean():.3f}")
    print(f"     Median accuracy: {df_base_summary['accuracy'].median():.3f}")
    print(f"     Min accuracy: {df_base_summary['accuracy'].min():.3f}")
    print(f"     Max accuracy: {df_base_summary['accuracy'].max():.3f}")
    
    if dry_run:
        print(f"\n[DRY RUN] Would update datasets but not pushing to hub.")
        return df_cue_summary, df_base_summary
    
    # Save to HuggingFace
    if push_to_hub:
        print(f"\n6. Pushing updated summaries to HuggingFace...")
        
        cue_dataset = Dataset.from_pandas(df_cue_summary.reset_index(drop=True))
        cue_dataset.push_to_hub(cue_summary_dataset)
        print(f"   ✓ Pushed {cue_summary_dataset}")
        
        base_dataset = Dataset.from_pandas(df_base_summary.reset_index(drop=True))
        base_dataset.push_to_hub(base_summary_dataset)
        print(f"   ✓ Pushed {base_summary_dataset}")
        
        print(f"\n✓ Successfully updated both summary datasets!")
    else:
        print(f"\n6. [SKIP] Not pushing to hub (use --push_to_hub to upload)")
        print(f"   Updated DataFrames are returned for local inspection")
    
    return df_cue_summary, df_base_summary


def main():
    parser = argparse.ArgumentParser(
        description="Add accuracy metrics to summary tables from long datasets"
    )
    parser.add_argument(
        "--cue-long",
        type=str,
        default="yulia-volkova/mmlu-chua-cue-long",
        help="HuggingFace dataset name for cue long rollouts",
    )
    parser.add_argument(
        "--base-long",
        type=str,
        default="yulia-volkova/mmlu-chua-base-long",
        help="HuggingFace dataset name for base long rollouts",
    )
    parser.add_argument(
        "--cue-summary",
        type=str,
        default="yulia-volkova/mmlu-chua-cue-summary",
        help="HuggingFace dataset name for cue summary",
    )
    parser.add_argument(
        "--base-summary",
        type=str,
        default="yulia-volkova/mmlu-chua-base-summary",
        help="HuggingFace dataset name for base summary",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push updated datasets to HuggingFace Hub",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done without making changes",
    )
    
    args = parser.parse_args()
    
    add_accuracy_to_summaries(
        cue_long_dataset=args.cue_long,
        base_long_dataset=args.base_long,
        cue_summary_dataset=args.cue_summary,
        base_summary_dataset=args.base_summary,
        push_to_hub=args.push_to_hub,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

