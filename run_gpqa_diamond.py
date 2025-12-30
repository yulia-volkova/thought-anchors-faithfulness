"""
Pipeline to generate rollout datasets for GPQA-Diamond.

Creates the same dataset structure as MMLU:
- cue-summary, cue-long (with Stanford professor cue)
- base-summary, base-long (without cue)
- no-reasoning-summary, no-reasoning-long

Uses prepare_gpqa_dataset.py for dataset construction.
"""

import os
import pandas as pd
from hf_utils import load_hf_as_df
from prepare_gpqa_dataset import prepare_gpqa_dataset
import argparse
from run_cued_uncued import (
    run_rollouts,
    save_as_hf_dataset,
)
from run_no_reasoning import (
    run_no_reasoning_rollouts,
    save_to_hf as save_no_reasoning_to_hf,
)


PREPARED_HF_DATASET = "yulia-volkova/gpqa-diamond-cued-prepared"


def run_gpqa_pipeline(
    use_prepared_hf: bool = True,  # Use already prepared HF dataset
    prepared_hf_id: str = PREPARED_HF_DATASET,
    source: str = "fingertap",  # "idavidrein" or "fingertap" 
    idavidrein_csv_path: str = "gpqa_diamond.csv",
    # Generation params
    num_responses: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    model: str = "deepseek-ai/deepseek-r1-distill-qwen-14b",
    seed: int = 42,
    # Output 
    push_to_hub: bool = False,
    base_repo_id: str = "yulia-volkova/gpqa-diamond",
    output_suffix: str = "",  # Optional suffix for output filenames (e.g., "_8192_mt")

    run_reasoning: bool = True,
    run_no_reasoning: bool = True,
):
    """
    Run the complete GPQA-Diamond pipeline.
    
    By default, loads prepared dataset from HuggingFace.
    Set use_prepared_hf=False to prepare from scratch.
    
    Generates:
    - cue-summary, cue-long 
    - base-summary, base-long 
    - no-reasoning-summary, no-reasoning-long
    """
    
    print("1: Loading GPQA-Diamond dataset")
    
    if use_prepared_hf:
        print(f"Loading prepared dataset from: {prepared_hf_id}")
        df = load_hf_as_df(prepared_hf_id, split="train")
        print(f"Loaded {len(df)} problems")
    else:
        print(f"Preparing dataset from source: {source}")
        df = prepare_gpqa_dataset(
            source=source,
            idavidrein_csv_path=idavidrein_csv_path,
            seed=seed,
            save_csv=True,
            output_dir="rollout_outputs/gpqa",
            output_filename="gpqa_prepared.csv",
        )
    
    results = {"df_prepared": df}
    
    if run_reasoning:
        print("\n" + "=" * 60)
        print("STEP 2: Running cued + base rollouts")
        print("=" * 60)
        
        df_for_rollouts = df.rename(columns={"question_reasoning": "question"})
        
        df_cue, df_base, df_cue_long, df_base_long = run_rollouts(
            df=df_for_rollouts,
            num_responses=num_responses,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            provider="local",
            model=model,
            max_retries=6,
            tokens_target=("ĠWait",),
        )
        
        # Save locally
        os.makedirs("rollout_outputs/gpqa", exist_ok=True)
        df_cue.to_csv(f"rollout_outputs/gpqa/df_cue_summary{output_suffix}.csv", index=False)
        df_base.to_csv(f"rollout_outputs/gpqa/df_base_summary{output_suffix}.csv", index=False)
        df_cue_long.to_csv(f"rollout_outputs/gpqa/df_cue_long{output_suffix}.csv", index=False)
        df_base_long.to_csv(f"rollout_outputs/gpqa/df_base_long{output_suffix}.csv", index=False)
        print(f"\nSaved reasoning CSVs to rollout_outputs/gpqa/ with suffix '{output_suffix}'")
        
        results.update({
            "df_cue": df_cue,
            "df_base": df_base,
            "df_cue_long": df_cue_long,
            "df_base_long": df_base_long,
        })
        
        # Save to HF
        if push_to_hub:
            save_as_hf_dataset(
                df_cue,
                df_base,
                df_cue_long,
                df_base_long,
                base_repo_id=base_repo_id,
                push_to_hub=True,
            )
    
    if run_no_reasoning:
        ("\n" + "=" * 60)
        print("STEP 3: Running no-reasoning rollouts")

        
        df_for_no_reasoning = df[["pi", "question_no_reasoning", "gt_answer", "cue_answer"]].copy()
        df_for_no_reasoning = df_for_no_reasoning.rename(columns={"question_no_reasoning": "question"})
        
        df_no_reasoning_summary, df_no_reasoning_long = run_no_reasoning_rollouts(
            df=df_for_no_reasoning,
            num_responses=num_responses,
            temperature=temperature,
            top_p=top_p,
            max_tokens=10,  # Increased to reduce \boxed{} incomplete outputs
            model=model,
            prompt_already_formatted=True,  # question_no_reasoning is already formatted
        )
        
        # Save locally
        df_no_reasoning_summary.to_csv(f"rollout_outputs/gpqa/df_no_reasoning_summary{output_suffix}.csv", index=False)
        df_no_reasoning_long.to_csv(f"rollout_outputs/gpqa/df_no_reasoning_long{output_suffix}.csv", index=False)
        print(f"\nSaved no-reasoning CSVs to rollout_outputs/gpqa/ with suffix '{output_suffix}'")
        
        results.update({
            "df_no_reasoning_summary": df_no_reasoning_summary,
            "df_no_reasoning_long": df_no_reasoning_long,
        })
        
    
        if push_to_hub:
            save_no_reasoning_to_hf(
                df_summary=df_no_reasoning_summary,
                df_long=df_no_reasoning_long,
                summary_repo="yulia-volkova/gpqa-diamond-no-reasoning-summary",
                long_repo="yulia-volkova/gpqa-diamond-no-reasoning-long",
                push_to_hub=True,
            )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="GPQA-Diamond rollouts pipeline")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push results to HuggingFace Hub"
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Skip reasoning rollouts (cued/base)"
    )
    parser.add_argument(
        "--skip-no-reasoning",
        action="store_true",
        help="Skip no-reasoning rollouts"
    )
    args = parser.parse_args()
    
    SOURCE = "huggingface"  # "local" or "huggingface"
    CSV_PATH = "gpqa_diamond.csv"  
    NUM_RESPONSES = 20
    MAX_TOKENS = 8192
    SEED = 42
    BASE_REPO_ID = "yulia-volkova/gpqa-diamond"
    OUTPUT_SUFFIX = "_8192_mt" 
    
    MODEL = "deepseek-ai/deepseek-r1-distill-qwen-14b"
    TEMPERATURE = 0.7
    TOP_P = 0.95
    
    results = run_gpqa_pipeline(
        use_prepared_hf=(SOURCE == "huggingface"),
        source="fingertap",
        idavidrein_csv_path=CSV_PATH,
        num_responses=NUM_RESPONSES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        model=MODEL,
        seed=SEED,
        push_to_hub=args.push_to_hub,
        base_repo_id=BASE_REPO_ID,
        output_suffix=OUTPUT_SUFFIX,
        run_reasoning=not args.skip_reasoning,
        run_no_reasoning=not args.skip_no_reasoning,
    )
