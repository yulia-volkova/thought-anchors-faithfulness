import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from hf_utils import load_hf_as_df


def main():
    parser = argparse.ArgumentParser(
        description="Create HF dataset with rollouts for top problems"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="selected_problems_2.json",
    )
    parser.add_argument(
        "--cue_long_dataset",
        type=str,
        default="yulia-volkova/mmlu-chua-cue-long",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Path to save as CSV file (e.g., selected_rollouts.csv)",
    )
    args = parser.parse_args()

    # 1. Load the good problems JSON
    print(f"Loading good problems from {args.input_json}...")
    with open(args.input_json, "r") as f:
        data = json.load(f)
    
    top_faithful = data.get("top_faithful", [])
    top_unfaithful = data.get("top_unfaithful", [])
    top_mixed = data.get("top_mixed", [])
    
    print(f"Found {len(top_faithful)} top faithful problems")
    print(f"Found {len(top_unfaithful)} top unfaithful problems")
    print(f"Found {len(top_mixed)} top mixed problems")

    # 2. Extract pi values
    pi_faithful = [r["pi"] for r in top_faithful]
    pi_unfaithful = [r["pi"] for r in top_unfaithful]
    pi_mixed = [r["pi"] for r in top_mixed]
    
    all_top_pis = set(pi_faithful + pi_unfaithful + pi_mixed)
    print(f"\nTotal unique problems: {len(all_top_pis)}")
    print(f"  - Faithful: {len(pi_faithful)}")
    print(f"  - Unfaithful: {len(pi_unfaithful)}")
    print(f"  - Mixed: {len(pi_mixed)}")

    # 3. Load cue-long dataset
    print(f"\nLoading rollouts from {args.cue_long_dataset}...")
    df_long = load_hf_as_df(args.cue_long_dataset)
    print(f"Loaded {len(df_long)} total rollouts")

    # 4. Filter to get rollouts for top problems
    df_top_rollouts = df_long[df_long["pi"].isin(all_top_pis)].copy()
    print(f"Filtered to {len(df_top_rollouts)} rollouts for top problems")

    # 5. Drop unwanted columns -- coming from chua csv
    columns_to_drop = [
        "original_answer",
        "judge_extracted_evidence",
        "cued_raw_response",
        "model",
        "ĠWait_count",
        "ĠWait_p",
    ]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_top_rollouts.columns]
    if existing_cols_to_drop:
        df_top_rollouts.drop(columns=existing_cols_to_drop, inplace=True)
        print(f"Dropped {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")

    # 6. Add category column
    def get_category(pi_val):
        if pi_val in pi_faithful:
            return "faithful"
        elif pi_val in pi_unfaithful:
            return "unfaithful"
        elif pi_val in pi_mixed:
            return "mixed"
        else:
            return None
    
    df_top_rollouts["category"] = df_top_rollouts["pi"].apply(get_category)
    
    print("\nRollout breakdown by category:")
    for cat in ["faithful", "unfaithful", "mixed"]:
        count = len(df_top_rollouts[df_top_rollouts["category"] == cat])
        print(f"  - {cat}: {count} rollouts")

    # 7. Create dataset
    dataset = Dataset.from_pandas(df_top_rollouts.reset_index(drop=True))
    
    if args.output_repo:
        hf_repo_id = args.output_repo
    else:
        # Generate from input_json filename
        input_path = Path(args.input_json)
        repo_name = "selected_problems_mmlu_professor_cue"
        hf_repo_id = f"yulia-volkova/{repo_name}-rollouts"
    
    # Save as CSV if requested
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_top_rollouts.to_csv(csv_path, index=False)
        print(f"\n✓ Saved CSV to: {csv_path}")

    if args.push_to_hub:
        print(f"\nPushing {len(df_top_rollouts)} rollouts to {hf_repo_id}...")
        dataset.push_to_hub(hf_repo_id)
        print(f"✓ Pushed to Hugging Face: {hf_repo_id}")
    elif not args.save_csv:
        # Only save to disk if not pushing to hub and not saving CSV
        local_dir = f"rollout_outputs/{Path(hf_repo_id).name}"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(local_dir)
        print(f"\n✓ Saved dataset locally to: {local_dir}")
        print(f"  (Use --push_to_hub to push to Hugging Face)")

    print("\n=== Summary ===")
    print(f"Total rollouts: {len(df_top_rollouts)}")
    print(f"  - Faithful problems: {len(pi_faithful)} × ~20 rollouts = {len(df_top_rollouts[df_top_rollouts['category'] == 'faithful'])} rows")
    print(f"  - Unfaithful problems: {len(pi_unfaithful)} × ~20 rollouts = {len(df_top_rollouts[df_top_rollouts['category'] == 'unfaithful'])} rows")
    print(f"  - Mixed problems: {len(pi_mixed)} × ~20 rollouts = {len(df_top_rollouts[df_top_rollouts['category'] == 'mixed'])} rows")


if __name__ == "__main__":
    main()

