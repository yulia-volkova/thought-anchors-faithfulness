"""
Script to filter HuggingFace datasets by removing rows where cue_answer == gt_answer.
Downloads, filters, and re-uploads the datasets.
"""

from datasets import load_dataset, Dataset
from huggingface_hub import delete_repo, HfApi


DATASETS = [
    "yulia-volkova/mmlu-chua-base-summary",
    "yulia-volkova/mmlu-chua-cue-summary",
    "yulia-volkova/mmlu-chua-cue-long",
    "yulia-volkova/mmlu-chua-base-long",
]


def filter_and_reupload(dataset_name: str, dry_run: bool = False):
    """
    Load dataset, filter out rows where cue_answer == gt_answer, and re-upload.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()
    
    before_count = len(df)
    print(f"Original row count: {before_count}")
    
    # Filter out rows where cue_answer == gt_answer
    df_filtered = df[df["cue_answer"] != df["gt_answer"]].copy()
    after_count = len(df_filtered)
    removed_count = before_count - after_count
    
    print(f"Rows removed (cue_answer == gt_answer): {removed_count}")
    print(f"Remaining row count: {after_count}")
    
    if removed_count == 0:
        print("No rows to remove, skipping re-upload.")
        return
    
    if dry_run:
        print("[DRY RUN] Would delete and re-upload dataset.")
        return
    
    # Convert back to Dataset
    ds_filtered = Dataset.from_pandas(df_filtered, preserve_index=False)
    
    # Delete original dataset
    print(f"Deleting original dataset: {dataset_name}...")
    try:
        delete_repo(repo_id=dataset_name, repo_type="dataset")
        print("Deleted successfully.")
    except Exception as e:
        print(f"Warning: Could not delete (may not exist or already deleted): {e}")
    
    # Re-upload filtered dataset
    print(f"Uploading filtered dataset to: {dataset_name}...")
    ds_filtered.push_to_hub(dataset_name, private=False)
    print("Upload complete!")
    
    return {
        "dataset": dataset_name,
        "before": before_count,
        "after": after_count,
        "removed": removed_count,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter HF datasets by removing rows where cue_answer == gt_answer"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done without making changes",
    )
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")
    
    results = []
    for dataset_name in DATASETS:
        try:
            result = filter_and_reupload(dataset_name, dry_run=args.dry_run)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['dataset']}: {r['before']} -> {r['after']} (removed {r['removed']})")
    
    if not results:
        print("  No datasets were modified.")


if __name__ == "__main__":
    main()



