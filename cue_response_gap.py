"""
Compute cue_response_gap from base and cue summary tables.

cue_response_gap = cue_match_cue - cue_match_base

where:
    cue_match_base = p(answer = x|no hint)
    cue_match_cue  = p(answer = x|hint = x)
"""

import pandas as pd


def prepare_merged_data(
    df_base: pd.DataFrame,
    df_cue: pd.DataFrame,
) -> pd.DataFrame:

    different_columns = ["gt_match", "cue_match", "other_match", "valid_responses"]
    merge_keys = ["pi"]
    
    common_columns = set(df_base.columns) & set(df_cue.columns)
    
    identical_columns = [col for col in common_columns 
                        if col not in different_columns and col not in merge_keys]
    
    # Check that identical columns have the same values for each merge key
    print(f"Checking {len(identical_columns)} columns for consistency between base and cue (by {merge_keys})...")
    for col in identical_columns:
        merged_df_to_check = pd.merge(
            df_base[merge_keys + [col]].rename(columns={col: f"{col}_base"}),
            df_cue[merge_keys + [col]].rename(columns={col: f"{col}_cue"}),
            on=merge_keys,
            how="inner",
        )
        
        # Check if values differ
        if not merged_df_to_check[f"{col}_base"].equals(merged_df_to_check[f"{col}_cue"]):
            mismatches = merged_df_to_check[merged_df_to_check[f"{col}_base"] != merged_df_to_check[f"{col}_cue"]]
            if not mismatches.empty:
                mismatch_info = ", ".join([f"{key}={mismatches.iloc[0][key]}" for key in merge_keys])
                raise ValueError(
                    f"Column '{col}' has different values between base and cue. "
                    f"First mismatch at {mismatch_info}: "
                    f"base={mismatches.iloc[0][f'{col}_base']}, "
                    f"cue={mismatches.iloc[0][f'{col}_cue']}"
                )
    
    print("All identical columns match between base and cue")
    
    # Merge strategy:
    # 1. Start with df_base (contains all common columns)
    # 2. Add only the diff_columns from df_cue with _cue suffix
    # 3. Rename diff_columns in df_base to have _base suffix
    
    # Prepare df_base: rename diff_columns to have _base suffix
    df_base_renamed = df_base.copy()
    rename_base = {col: f"{col}_base" for col in different_columns if col in df_base_renamed.columns}
    df_base_renamed.rename(columns=rename_base, inplace=True)
    
    # Prepare df_cue: select only diff_columns and merge keys, rename diff_columns to have _cue suffix
    df_cue_different = df_cue[merge_keys + [col for col in different_columns if col in df_cue.columns]].copy()
    rename_cue = {col: f"{col}_cue" for col in different_columns if col in df_cue_different.columns}
    df_cue_different.rename(columns=rename_cue, inplace=True)
    
    # Merge: df_base_renamed has all columns (identical + diff with _base), 
    # df_cue_different has only merge_keys + diff_columns with _cue suffix
    merged = pd.merge(
        df_base_renamed,
        df_cue_different,
        on=merge_keys,
        how="inner",
    )
    
    if merged.empty:
        raise ValueError(
            "Merged base/cue summary is empty. Check that merge keys align between datasets."
        )
    
    # Check that merged table has the same number of rows as original tables
    if len(merged) != len(df_base):
        raise ValueError(
            f"Merged table has {len(merged)} rows but df_base has {len(df_base)} rows. "
        )
    if len(merged) != len(df_cue):
        raise ValueError(
            f"Merged table has {len(merged)} rows but df_cue has {len(df_cue)} rows. "
        )
    
    print(f"✓ Merge successful: {len(merged)} rows (matches original tables)")

    return merged


def compute_cue_response_gap(
    df_base: pd.DataFrame,
    df_cue: pd.DataFrame
) -> pd.DataFrame:
    merged = prepare_merged_data(df_base, df_cue)
    
    merged["cue_response_gap"] = (
        merged["cue_match_cue"] - merged["cue_match_base"]
    )

    return merged

