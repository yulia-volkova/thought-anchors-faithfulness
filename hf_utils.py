from datasets import load_dataset, Dataset
import pandas as pd


def load_hf_as_df(dataset_name: str, split: str = "train") -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    return ds.to_pandas()


def push_df_to_hf(df: pd.DataFrame, repo_id: str) -> None:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset.push_to_hub(repo_id)
    print(f"Pushed to HuggingFace: {repo_id}")

