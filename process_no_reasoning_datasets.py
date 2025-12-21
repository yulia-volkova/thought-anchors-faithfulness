"""Process no-reasoning datasets to add clean_question and most_common_given_answer columns."""

import pandas as pd
from collections import Counter
from hf_utils import load_hf_as_df, push_df_to_hf
from run_no_reasoning import format_no_reasoning_prompt

def add_clean_question_column(df_long: pd.DataFrame) -> pd.DataFrame:
    """Add clean_question_for_no_reasoning column to long dataset."""
    df_long = df_long.copy()
    
    print("Adding clean_question_for_no_reasoning column...")
    df_long["clean_question_for_no_reasoning"] = df_long["question"].apply(format_no_reasoning_prompt)
    
    return df_long


def add_most_common_answer_column(df_summary: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    """Add most_common_given_answer column to summary dataset based on long dataset."""
    df_summary = df_summary.copy()
    
    print("Calculating most_common_given_answer from long dataset...")
    
    # Group by pi and count answers
    answer_counts = {}
    for pi, group in df_long.groupby("pi"):
        # Get all non-null answers for this problem
        answers = group["answer"].dropna().tolist()
        if answers:
            # Find most common answer
            counter = Counter(answers)
            most_common = counter.most_common(1)[0][0]
            answer_counts[pi] = most_common
        else:
            answer_counts[pi] = None
    
    # Add column to summary
    df_summary["most_common_given_answer"] = df_summary["pi"].map(answer_counts)
    
    return df_summary


def main():
    # Load datasets
    print("Loading datasets from HuggingFace...")
    df_long = load_hf_as_df("yulia-volkova/mmlu-chua-no-reasoning-long")
    df_summary = load_hf_as_df("yulia-volkova/mmlu-chua-no-reasoning-summary")
    
    print(f"Loaded long dataset: {len(df_long)} rows")
    print(f"Loaded summary dataset: {len(df_summary)} rows")
    
    # Process long dataset
    df_long_processed = add_clean_question_column(df_long)
    
    # Process summary dataset
    df_summary_processed = add_most_common_answer_column(df_summary, df_long_processed)
    
    # Save locally
    import os
    os.makedirs("rollout_outputs", exist_ok=True)
    
    df_long_processed.to_csv("rollout_outputs/df_no_reasoning_long_processed.csv", index=False)
    df_summary_processed.to_csv("rollout_outputs/df_no_reasoning_summary_processed.csv", index=False)
    print("\nSaved processed CSVs to rollout_outputs/")
    
    # Print some stats
    print("\n=== Stats ===")
    print(f"Long dataset columns: {list(df_long_processed.columns)}")
    print(f"Summary dataset columns: {list(df_summary_processed.columns)}")
    print(f"\nMost common answers distribution:")
    print(df_summary_processed["most_common_given_answer"].value_counts())
    
    # Push to HuggingFace
    print("\nPushing processed datasets to HuggingFace...")
    push_df_to_hf(df_long_processed, "yulia-volkova/mmlu-chua-no-reasoning-long")
    push_df_to_hf(df_summary_processed, "yulia-volkova/mmlu-chua-no-reasoning-summary")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()



