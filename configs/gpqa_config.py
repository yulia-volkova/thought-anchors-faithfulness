
CONFIG = {
    "name": "gpqa",
    "source": "huggingface",  # "huggingface" or "local"
    "prepared_source_data_hf_id": "yulia-volkova/gpqa-diamond-cued-prepared",
    "idavidrein_csv_path": "data/gpqa_diamond.csv",
    "fingertap_source": "fingertap",  # For local preparation
    
    "model": "deepseek-ai/deepseek-r1-distill-qwen-14b",
    "temperature": 0.7,
    "top_p": 0.95,
    
    "max_tokens": 8192,  # GPQA needs longer reasoning
    "num_responses": 20,
    "max_retries": 6,
    "tokens_target": ("ĠWait",),
    
    
    "output_dir": "rollout_outputs/gpqa",
    "output_suffix": "_8192_mt",
    
    "hf_base_repo_id": "yulia-volkova/gpqa-diamond",
    
    # Pipeline defaults 
    "run_reasoning": True,
    "run_no_reasoning": True,
    
    # GPQA-specific: no-reasoning max tokens 
    "no_reasoning_max_tokens": 10,
}
