"""
Script to remove duplicate problems from HuggingFace datasets based on question_with_cue.
Filters duplicates, deletes original datasets, and uploads cleaned versions.
"""

import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import delete_repo, HfApi
import pandas as pd


def filter_duplicate_problems(df: pd.DataFrame, key_column: str = "question_with_cue") -> pd.DataFrame:

    before_count = len(df)
    duplicate_counts = df.groupby(key_column).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} {key_column} values with duplicates:")
        print(f"  Total duplicate rows: {duplicates.sum() - len(duplicates)}")
        filtered_df = df.drop_duplicates(subset=[key_column], keep="first")
        after_count = len(filtered_df)
        print(f"Filtered out {before_count - after_count} duplicate rows (kept first occurrence)")
    else:
        filtered_df = df.copy()
        print(f"No duplicate {key_column} values found")
    
    return filtered_df


def remove_duplicates_and_reupload(
    dataset_name: str,
    key_column: str = "question_with_cue",
    dry_run: bool = False,
):
    """
    Load dataset, remove duplicates based on key_column, and re-upload.
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
    
    # Check if key_column exists
    if key_column not in df.columns:
        print(f"Warning: Column '{key_column}' not found in dataset. Skipping...")
        return None
    
    # Filter duplicates
    df_filtered = filter_duplicate_problems(df, key_column=key_column)
    after_count = len(df_filtered)
    removed_count = before_count - after_count
    
    if removed_count == 0:
        print("No duplicates to remove, skipping re-upload.")
        return {
            "dataset": dataset_name,
            "before": before_count,
            "after": after_count,
            "removed": removed_count,
        }
    
    if dry_run:
        print(f"[DRY RUN] Would delete and re-upload dataset.")
        return {
            "dataset": dataset_name,
            "before": before_count,
            "after": after_count,
            "removed": removed_count,
        }
    
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


def filter_long_dataset_by_pis(
    dataset_name: str,
    unique_pis: set,
    dry_run: bool = False,
):
    """
    Filter long dataset to only include a unique question with cue associated with unique pi value 
    """
    print(f"\n{'='*60}")
    print(f"Processing long dataset: {dataset_name}")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()
    
    before_count = len(df)
    print(f"Original row count: {before_count}")
    

    # Filter to only include unique pi values
    df_filtered = df[df["pi"].isin(unique_pis)].copy()
    after_count = len(df_filtered)
    removed_count = before_count - after_count
    
    print(f"Filtered to {after_count} rows (removed {removed_count} rows for duplicate pi values)")
    
    if removed_count == 0:
        print("No rows to remove, skipping re-upload.")
        return {
            "dataset": dataset_name,
            "before": before_count,
            "after": after_count,
            "removed": removed_count,
        }
    
    if dry_run:
        print(f"[DRY RUN] Would delete and re-upload dataset.")
        return {
            "dataset": dataset_name,
            "before": before_count,
            "after": after_count,
            "removed": removed_count,
        }
    
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
    parser = argparse.ArgumentParser(
        description="Remove duplicate problems from HF datasets based on question_with_cue"
    )
    parser.add_argument(
        "--summary-datasets",
        nargs="+",
        default=[
            "yulia-volkova/mmlu-chua-base-summary",
            "yulia-volkova/mmlu-chua-cue-summary",
        ],
        help="List of HuggingFace summary dataset names to process",
    )
    parser.add_argument(
        "--long-datasets",
        nargs="+",
        default=[
            "yulia-volkova/mmlu-chua-base-long",
            "yulia-volkova/mmlu-chua-cue-long",
        ],
        help="List of HuggingFace long dataset names to filter by unique pis",
    )
    parser.add_argument(
        "--key-column",
        type=str,
        default="question_with_cue",
        help="Column name to use for duplicate detection (default: question_with_cue)",
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
    
    # Step 1: Process summary datasets and get unique pi values
    print("="*60)
    print("STEP 1: Processing Summary Datasets")
    print("="*60)
    
    summary_dfs = {}
    for dataset_name in args.summary_datasets:
        try:
            # Load dataset
            ds = load_dataset(dataset_name, split="train")
            df = ds.to_pandas()
            
            before_count = len(df)
            
            # Filter duplicates
            df_filtered = filter_duplicate_problems(df, key_column=args.key_column)
            after_count = len(df_filtered)
            removed_count = before_count - after_count
            
            # Store filtered dataframe for getting unique pis
            summary_dfs[dataset_name] = df_filtered
            
            # Re-upload if not dry-run and there were duplicates removed
            if not args.dry_run and removed_count > 0:
                ds_filtered = Dataset.from_pandas(df_filtered, preserve_index=False)
                
                print(f"Deleting original dataset: {dataset_name}...")
                try:
                    delete_repo(repo_id=dataset_name, repo_type="dataset")
                    print("Deleted successfully.")
                except Exception as e:
                    print(f"Warning: Could not delete: {e}")
                
                print(f"Uploading filtered dataset to: {dataset_name}...")
                ds_filtered.push_to_hub(dataset_name, private=False)
                print("Upload complete!")
            
            results.append({
                "dataset": dataset_name,
                "before": before_count,
                "after": after_count,
                "removed": removed_count,
            })
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 2: Get unique pi values from summary datasets
    unique_pis = None
    if summary_dfs:
        print("\n" + "="*60)
        print("STEP 2: Getting Unique PI Values")
        print("="*60)
        
        # Get unique pis from each summary dataset
        all_unique_pis = []
        for dataset_name, df in summary_dfs.items():
            pis = set(df["pi"].unique())
            all_unique_pis.append(pis)
            print(f"  {dataset_name}: {len(pis)} unique pi values")
        
        # Get intersection of all unique pis (if multiple datasets) or use single dataset
        if len(all_unique_pis) > 1:
            unique_pis = set.intersection(*all_unique_pis)
            print(f"  Valid pi values (intersection): {len(unique_pis)}")
        else:
            unique_pis = all_unique_pis[0]
            print(f"  Valid pi values: {len(unique_pis)}")
    
    # Step 3: Filter long datasets
    if args.long_datasets and unique_pis:
            print("\n" + "="*60)
            print("STEP 3: Filtering Long Datasets")
            print("="*60)
            
            for dataset_name in args.long_datasets:
                try:
                    result = filter_long_dataset_by_pis(
                        dataset_name,
                        unique_pis,
                        dry_run=args.dry_run,
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"ERROR processing {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['dataset']}: {r['before']} -> {r['after']} (removed {r['removed']})")
    
    if not results:
        print("  No datasets were modified.")
    
    total_removed = sum(r['removed'] for r in results)
    if total_removed > 0:
        print(f"\nTotal rows removed: {total_removed}")


if __name__ == "__main__":
    main()

