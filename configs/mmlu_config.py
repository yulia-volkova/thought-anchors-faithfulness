
CONFIG = {
    "name": "mmlu",
    "source": "chua_csv",  # "chua_csv" or "huggingface"
    "chua_csv_path": "data/Chua_faithfulness_results.csv",
    "cue_type": "Professor",
    "conditions": ["itc_failure", "itc_success"],
    
    "model": "deepseek-ai/deepseek-r1-distill-qwen-14b",
    "temperature": 0.7,
    "top_p": 0.95,
    
    "max_tokens": 2048,  # MMLU questions are shorter
    "num_responses": 20,
    "max_retries": 6,
    "tokens_target": ("ĠWait",),
    
    "output_dir": "rollout_outputs/mmlu",
    "output_suffix": "",
    
 
    # Pipeline defaults
    "run_reasoning": True,
    "run_no_reasoning": True,
    
    "no_reasoning_max_tokens": 10,
}
