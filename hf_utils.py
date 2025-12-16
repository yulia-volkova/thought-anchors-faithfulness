from datasets import load_dataset
import pandas as pd


def load_hf_as_df(dataset_name: str, split: str = "train") -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    return ds.to_pandas()

