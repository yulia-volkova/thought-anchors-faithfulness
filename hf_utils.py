from datasets import load_dataset, Dataset
import pandas as pd


def load_hf_as_df(dataset_name: str, split: str = "train") -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    return ds.to_pandas()


def push_df_to_hf(df: pd.DataFrame, repo_id: str) -> None:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset.push_to_hub(repo_id)
    print(f"Pushed to HuggingFace: {repo_id}")


def push_rollouts_to_hf(dataframes: dict, base_repo_id: str) -> None:
    """
    Push multiple dataframes to HuggingFace Hub.
    
    Args:
        dataframes: Dict mapping suffix to dataframe, e.g. {"cue_summary": df_cue, ...}
        base_repo_id: Base repo ID, e.g. "yulia-volkova/mmlu-chua-rollouts"
    
    Creates repos like: {base_repo_id}-{suffix}
    """
    print(f"\nPushing {len(dataframes)} datasets to HuggingFace Hub...")
    for suffix, df in dataframes.items():
        push_df_to_hf(df, f"{base_repo_id}-{suffix}")

