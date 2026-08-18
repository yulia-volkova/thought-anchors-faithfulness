"""
    python run_pipeline.py --dataset mmlu
    python run_pipeline.py --dataset gpqa
    python run_pipeline.py --dataset gpqa --push-to-hub
    python run_pipeline.py --dataset mmlu --skip-no-reasoning
"""

import os
import argparse
import importlib
import pandas as pd

from run_cued_uncued import run_rollouts, load_preprocessed_chua_csv
from run_no_reasoning import run_no_reasoning_rollouts
from hf_utils import push_rollouts_to_hf


def load_config(dataset: str) -> dict:
    try:
        config_module = importlib.import_module(f"configs.{dataset}_config")
        return config_module.CONFIG
    except ModuleNotFoundError:
        raise ValueError(f"No config found for dataset: {dataset}. "
                        f"Expected configs/{dataset}_config.py")


def load_mmlu_data(config: dict) -> pd.DataFrame:
    """Load and preprocess MMLU data from Chua CSV."""
    df = load_preprocessed_chua_csv(
        cue_type=config["cue_type"],
        cond=config["conditions"],
    )
    return df


def load_gpqa_data(config: dict) -> pd.DataFrame:
    """Load GPQA data from HuggingFace or prepare locally."""
    from hf_utils import load_hf_as_df
    from prepare_gpqa_dataset import prepare_gpqa_dataset
    
    if config["source"] == "huggingface":
        print(f"Loading prepared dataset from: {config['prepared_source_data_hf_id']}")
        df = load_hf_as_df(config["prepared_source_data_hf_id"], split="train")
        # Rename for compatibility with run_rollouts
        df = df.rename(columns={"question_reasoning": "question"})
    else:
        print(f"Preparing dataset from source: {config['fingertap_source']}")
        df = prepare_gpqa_dataset(
            source=config["fingertap_source"],
            idavidrein_csv_path=config["idavidrein_csv_path"],
            seed=42,
            save_csv=True,
            output_dir=config["output_dir"],
            output_filename="gpqa_prepared.csv",
        )
        df = df.rename(columns={"question_reasoning": "question"})
    
    return df


def load_dataset(config: dict) -> pd.DataFrame:
    dataset_name = config["name"]
    if dataset_name == "mmlu":
        return load_mmlu_data(config)
    elif dataset_name == "gpqa":
        return load_gpqa_data(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_pipeline(
    dataset: str,
    push_to_hub: bool = False,
    skip_reasoning: bool = False,
    skip_no_reasoning: bool = False,
):
    """
        dataset: Dataset name ("mmlu" or "gpqa")
        push_to_hub: Whether to push results to HuggingFace Hub
        skip_reasoning: Skip cued/base reasoning rollouts
        skip_no_reasoning: Skip no-reasoning rollouts
    """
    config = load_config(dataset)
    
    print("=" * 60)
    print(f"Running pipeline for: {config['name'].upper()}")
    print(f"Description: {config['description']}")
    print("=" * 60)
    print(f"Model: {config['model']}")
    print(f"Max tokens: {config['max_tokens']}")
    print(f"Num responses: {config['num_responses']}")
    print("=" * 60)

    df = load_dataset(config)
    print(f"Loaded {len(df)} problems")
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    results = {"df_prepared": df, "config": config}
    suffix = config.get("output_suffix", "")
    
    # Run reasoning rollouts (cued + base)
    if config.get("run_reasoning", True) and not skip_reasoning:
        print("\n" + "=" * 60)
        print("Running cued + base rollouts")
        print("=" * 60)
        
        df_cue, df_base, df_cue_long, df_base_long = run_rollouts(
            df=df,
            num_responses=config["num_responses"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            provider="local",
            model=config["model"],
            max_retries=config["max_retries"],
            tokens_target=config["tokens_target"],
        )
        
        output_dir = config["output_dir"]
        df_cue.to_csv(f"{output_dir}/df_cue_summary{suffix}.csv", index=False)
        df_base.to_csv(f"{output_dir}/df_base_summary{suffix}.csv", index=False)
        df_cue_long.to_csv(f"{output_dir}/df_cue_long{suffix}.csv", index=False)
        df_base_long.to_csv(f"{output_dir}/df_base_long{suffix}.csv", index=False)
        print(f"\nSaved reasoning CSVs to {output_dir}/")
        
        results.update({
            "df_cue": df_cue,
            "df_base": df_base,
            "df_cue_long": df_cue_long,
            "df_base_long": df_base_long,
        })
        
        if push_to_hub:
            push_rollouts_to_hf(
                dataframes={
                    "cue_summary": df_cue,
                    "base_summary": df_base,
                    "cue_long": df_cue_long,
                    "base_long": df_base_long,
                },
                base_repo_id=config["hf_base_repo_id"],
            )
    
    if config.get("run_no_reasoning", True) and not skip_no_reasoning:
        print("\n" + "=" * 60)
        print("Running no-reasoning rollouts")
        print("=" * 60)
        # Note: cue is not used for no-reasoning rollouts, just kept for consistency
        if "question_no_reasoning" in df.columns:
            df_for_no_reasoning = df[["pi", "question_no_reasoning", "gt_answer", "cue_answer"]].copy()
            df_for_no_reasoning = df_for_no_reasoning.rename(columns={"question_no_reasoning": "question"})
            prompt_already_formatted = True
        else:
            # For MMLU from Chua CSV, we need different handling
            df_for_no_reasoning = df[["pi", "question", "gt_answer", "cue_answer"]].copy()
            prompt_already_formatted = False
        
        no_reasoning_max_tokens = config.get("no_reasoning_max_tokens", 10)
        
        df_no_reasoning_summary, df_no_reasoning_long = run_no_reasoning_rollouts(
            df=df_for_no_reasoning,
            num_responses=config["num_responses"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=no_reasoning_max_tokens,
            model=config["model"],
            prompt_already_formatted=prompt_already_formatted,
        )
        
        # Save locally
        output_dir = config["output_dir"]
        df_no_reasoning_summary.to_csv(f"{output_dir}/df_no_reasoning_summary{suffix}.csv", index=False)
        df_no_reasoning_long.to_csv(f"{output_dir}/df_no_reasoning_long{suffix}.csv", index=False)
        print(f"\nSaved no-reasoning CSVs to {output_dir}/")
        
        results.update({
            "df_no_reasoning_summary": df_no_reasoning_summary,
            "df_no_reasoning_long": df_no_reasoning_long,
        })
        
        if push_to_hub:
            push_rollouts_to_hf(
                dataframes={
                    "no_reasoning_summary": df_no_reasoning_summary,
                    "no_reasoning_long": df_no_reasoning_long,
                },
                base_repo_id=config["hf_base_repo_id"],
            )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mmlu", "gpqa"]
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true"
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true"
    )
    parser.add_argument(
        "--skip-no-reasoning",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        dataset=args.dataset,
        push_to_hub=args.push_to_hub,
        skip_reasoning=args.skip_reasoning,
        skip_no_reasoning=args.skip_no_reasoning,
    )


if __name__ == "__main__":
    main()
