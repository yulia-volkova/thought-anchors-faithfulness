"""
Test script for GPQA pipeline with a single problem.
"""

from hf_utils import load_hf_as_df, push_df_to_hf
from run_cued_uncued import run_rollouts
from run_no_reasoning import run_no_reasoning_rollouts


def test_gpqa_pipeline(
    pi: int = 0,
    num_responses: int = 2,
    max_tokens: int = 8192,
    push_to_hub: bool = False,
    base_repo_id: str = "yulia-volkova/gpqa-diamond-test",
):
    
    # Load and filter to 1 problem
    print("Loading dataset...")
    df = load_hf_as_df("yulia-volkova/gpqa-diamond-cued-prepared", split="train")
    df_test = df[df["pi"] == pi].copy()
    print(f"Testing with pi={pi}")
    print(f"GT answer: {df_test['gt_answer'].iloc[0]}, Cue answer: {df_test['cue_answer'].iloc[0]}")
    
    # Preview prompts
    print("\n" + "=" * 60)
    print("REASONING PROMPT (first 500 chars):")
    print("=" * 60)
    print(df_test["question_reasoning"].iloc[0][:500])
    
    print("\n" + "=" * 60)
    print("NO-REASONING PROMPT (first 500 chars):")
    print("=" * 60)
    print(df_test["question_no_reasoning"].iloc[0][:500])
    
    # === Run reasoning rollouts ===
    print("\n" + "=" * 60)
    print(f"Running reasoning rollouts ({num_responses} responses)...")
    print("=" * 60)
    
    df_for_rollouts = df_test.rename(columns={"question_reasoning": "question"})
    
    df_cue, df_base, df_cue_long, df_base_long = run_rollouts(
        df=df_for_rollouts,
        num_responses=num_responses,
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        provider="local",
        model="deepseek-ai/deepseek-r1-distill-qwen-14b",
        max_retries=3,
    )
    
    print("\nCue summary:")
    print(df_cue[["pi", "gt_answer", "cue_answer", "gt_match", "cue_match"]].to_string())
    print("\nBase summary:")
    print(df_base[["pi", "gt_answer", "cue_answer", "gt_match", "cue_match"]].to_string())
    
    # === Run no-reasoning rollouts ===
    print("\n" + "=" * 60)
    print(f"Running no-reasoning rollouts ({num_responses} responses)...")
    print("=" * 60)
    
    df_no_reason = df_test[["pi", "question_no_reasoning", "gt_answer", "cue_answer"]].copy()
    df_no_reason = df_no_reason.rename(columns={"question_no_reasoning": "question"})
    
    df_nr_summary, df_nr_long = run_no_reasoning_rollouts(
        df=df_no_reason,
        num_responses=num_responses,
        temperature=0.7,
        top_p=0.95,
        max_tokens=3,
        model="deepseek-ai/deepseek-r1-distill-qwen-14b",
        prompt_already_formatted=True,
    )
    
    print("\nNo-reasoning summary:")
    print(df_nr_summary[["pi", "gt_answer", "accuracy", "n_valid_responses"]].to_string())
    
    # === Sample outputs ===
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    
    print("\n--- Cued rollout (first 300 chars) ---")
    print(df_cue_long["model_text"].iloc[0][:300])
    print(f"\nExtracted answer: {df_cue_long['answer'].iloc[0]}")
    
    print("\n--- Base rollout (first 300 chars) ---")
    print(df_base_long["model_text"].iloc[0][:300])
    print(f"\nExtracted answer: {df_base_long['answer'].iloc[0]}")
    
    print("\n--- No-reasoning rollout ---")
    print(f"Full output: {df_nr_long['model_text'].iloc[0]}")
    print(f"Extracted answer: {df_nr_long['answer'].iloc[0]}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    results = {
        "df_cue": df_cue,
        "df_base": df_base,
        "df_cue_long": df_cue_long,
        "df_base_long": df_base_long,
        "df_nr_summary": df_nr_summary,
        "df_nr_long": df_nr_long,
    }
    
    # Upload to HuggingFace if requested
    if push_to_hub:
        print("\n" + "=" * 60)
        print("Uploading to HuggingFace Hub...")
        print("=" * 60)
        
        datasets_to_upload = {
            "cue-summary": df_cue,
            "base-summary": df_base,
            "cue-long": df_cue_long,
            "base-long": df_base_long,
            "no-reasoning-summary": df_nr_summary,
            "no-reasoning-long": df_nr_long,
        }
        
        for name, df in datasets_to_upload.items():
            repo_id = f"{base_repo_id}-{name}"
            print(f"  Uploading {name} to {repo_id}...")
            push_df_to_hf(df, repo_id)
        
        print("\nAll datasets uploaded!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GPQA pipeline")
    parser.add_argument("--pi", type=int, default=0, help="Problem index to test")
    parser.add_argument("--num-responses", type=int, default=2, help="Number of rollouts")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for reasoning")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload results to HuggingFace")
    parser.add_argument("--base-repo-id", type=str, default="yulia-volkova/gpqa-diamond-test", 
                        help="Base repo ID for HuggingFace uploads")
    
    args = parser.parse_args()
    
    test_gpqa_pipeline(
        pi=args.pi,
        num_responses=args.num_responses,
        max_tokens=args.max_tokens,
        push_to_hub=args.push_to_hub,
        base_repo_id=args.base_repo_id,
    )

