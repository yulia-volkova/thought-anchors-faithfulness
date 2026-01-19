# reextract_answers.py

"""
Re-extract answers from existing rollouts using improved extraction logic.
This avoids re-running the expensive generation step.

Usage:
    python reextract_answers.py --input rollouts.parquet --output rollouts_fixed.parquet
"""

import argparse
import pandas as pd
from generate_rollouts import extract_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="rollouts.parquet")
    parser.add_argument("--output", type=str, default="rollouts_fixed.parquet")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    
    original_none = df["parsed_answer"].isna().sum()
    print(f"Original None count: {original_none} / {len(df)}")
    
    # Re-extract answers
    print("Re-extracting answers with improved logic...")
    df["parsed_answer"] = df["rollout_text"].apply(extract_answer)
    
    new_none = df["parsed_answer"].isna().sum()
    print(f"New None count: {new_none} / {len(df)}")
    print(f"Fixed: {original_none - new_none} rollouts")
    
    # Save
    df.to_parquet(args.output)
    print(f"\nSaved to {args.output}")
    
    # Show some stats
    print("\n--- Answer Distribution ---")
    print(df["parsed_answer"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

