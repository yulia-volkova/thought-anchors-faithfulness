#!/usr/bin/env python3
"""
Populate faithful_vs_unfaithful/ directories for existing attention analysis results.

1. Reusing the existing faithful rollout from cued/ (if it's faithful)
2. Generating only the missing unfaithful rollout

Usage:
    python populate_faithful_vs_unfaithful.py --dataset mmlu
    python populate_faithful_vs_unfaithful.py --dataset gpqa
    python populate_faithful_vs_unfaithful.py --dataset all
"""

import os
import sys
import json
import argparse
import shutil
import gc
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# CUDA memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-14b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Generation settings (must match original)
MAX_NEW_TOKENS_MMLU = 2048
MAX_NEW_TOKENS_GPQA = 8192
TEMPERATURE = 0.9
TOP_P = 0.95

# Paths
FINAL_DIR = "final"
MMLU_PROBLEMS_PATH = "data/selected_problems_mmlu.json"
GPQA_PROBLEMS_PATH = "data/selected_problems_gpqa-8192-mt.json"
MMLU_ROLLOUTS_CSV = "data/selected_rollouts_mmlu.csv"

# Cue patterns for detecting professor mentions
CUE_PATTERNS = [
    r"professor", r"stanford", r"iq\s*(?:of)?\s*130"
]


def clear_cuda_memory(verbose=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [CUDA Memory] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def mentions_cue(text, patterns=CUE_PATTERNS):
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in patterns)

def load_problems(dataset):
    if dataset == "mmlu":
        path = MMLU_PROBLEMS_PATH
    else:
        path = GPQA_PROBLEMS_PATH
    
    with open(path, "r") as f:
        data = json.load(f)
    
    pi_lookup = {}
    for cat in ["top_faithful", "top_unfaithful", "top_mixed"]:
        for p in data.get(cat, []):
            pi_lookup[p["pi"]] = p
    return pi_lookup

# Cache for MMLU rollouts CSV
_mmlu_rollouts_df = None

def load_mmlu_rollouts_csv():
    # Load MMLU rollouts CSV (cached)
    global _mmlu_rollouts_df
    if _mmlu_rollouts_df is None:
        import pandas as pd
        print("  Loading MMLU rollouts CSV...")
        _mmlu_rollouts_df = pd.read_csv(MMLU_ROLLOUTS_CSV)
    return _mmlu_rollouts_df

def get_existing_rollout_from_csv(pi, faithful=True, dataset="mmlu"):
    """Get an existing rollout text from CSV.
    
    Args:
        pi: Problem index
        faithful: If True, get a faithful (mentions professor) rollout.
                  If False, get an unfaithful one.
        dataset: 'mmlu' or 'gpqa' (only mmlu supported for now)
    
    Returns:
        dict with 'text' and 'response_idx', or None if not found
    """
    if dataset != "mmlu":
        return None  # Only MMLU has CSV with all rollouts
    
    df = load_mmlu_rollouts_csv()
    pi_rows = df[(df['pi'] == pi) & (df['condition'] == 'cue')]
    
    if len(pi_rows) == 0:
        return None
    
    # Filter by faithfulness (contains_professor is 'Yes' or 'No' string)
    if faithful:
        matching = pi_rows[pi_rows['contains_professor'].str.lower() == 'yes']
    else:
        matching = pi_rows[pi_rows['contains_professor'].str.lower() == 'no']
    
    if len(matching) == 0:
        return None
    
    # Get first matching rollout
    row = matching.iloc[0]
    return {
        'text': row['model_text'],
        'response_idx': int(row['response_idx'])
    }


model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="eager"  # Need eager for attention weights
        )
        model.eval()
        print("Model loaded!")
    return model, tokenizer


def generate_rollout(prompt, max_new_tokens, include_prompt=True):
    """Generate a single rollout with OOM handling."""
    import gc
    global model, tokenizer
    
    model, tokenizer = load_model()
    
    # Clear before generation
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # Get the device of the model's first parameter (handles device_map="auto")
        model_device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        full_ids = outputs[0].cpu()  # Move to CPU immediately
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        
        # Cleanup
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        if include_prompt:
            return {"ids": full_ids, "text": full_text, "prompt_len": input_len}
        else:
            response_ids = full_ids[input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            return {"ids": response_ids, "text": response_text, "prompt_len": 0}
            
    except torch.cuda.OutOfMemoryError:
        print("     ❌ OOM during generation - reloading model...")
        # Force cleanup and reload
        del model, tokenizer
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        return None  # Signal failure

MAX_SEQ_LEN_FOR_ATTENTION = 4096  # Limit to avoid OOM

def get_attention_weights(ids, max_len=MAX_SEQ_LEN_FOR_ATTENTION):
    import gc
    
    model, tokenizer = load_model()
    
    if isinstance(ids, list):
        ids = torch.tensor(ids)
    
    # Truncate if too long to avoid OOM
    original_len = len(ids)
    if len(ids) > max_len:
        print(f"  Truncating sequence from {original_len} to {max_len} tokens for attention")
        ids = ids[:max_len]
    
    # Get the device of the model's first parameter (handles device_map="auto")
    model_device = next(model.parameters()).device
    ids = ids.unsqueeze(0).to(model_device) if ids.dim() == 1 else ids.to(model_device)
    
    # Clear memory before heavy computation
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            outputs = model(ids, output_attentions=True, use_cache=False)
        
        attentions = []
        for layer_attn in outputs.attentions:
            attentions.append(layer_attn[0].cpu())  # Move to CPU immediately
        
        del outputs
        del ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return attentions, original_len
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"     OOM even with truncation. Clearing and returning None.")
        del ids
        gc.collect()
        torch.cuda.empty_cache()
        return None, original_len

def split_into_sentences(text):
    # Lazy loading: the import only happens if the function is called
    from anchors_utils import split_solution_into_chunks
    return split_solution_into_chunks(text)

def split_prompt_into_sentences(prompt_text):
    """Split prompt into chunks. Wrapper around anchors_utils.split_prompt_into_chunks."""
    from anchors_utils import split_prompt_into_chunks
    return split_prompt_into_chunks(prompt_text)

def get_sentence_token_ranges(text, sentences, tokenizer):
    """Get token ranges for each sentence."""
    from anchors_utils import get_chunk_ranges, get_chunk_token_ranges
    # First get character ranges, then convert to token ranges
    chunk_ranges = get_chunk_ranges(text, sentences)
    return get_chunk_token_ranges(text, chunk_ranges, tokenizer)

   
def save_rollout_data(rollout, save_dir, filename="rollout.json"):
    """Save rollout data to JSON, including token_ranges for visualization."""
    model, tokenizer = load_model()
    
    # Compute token_ranges if not present
    if "token_ranges" not in rollout and "sentences" in rollout:
        token_ranges = get_sentence_token_ranges(rollout["text"], rollout["sentences"], tokenizer)
        rollout["token_ranges"] = token_ranges
    
    data = {
        "text": rollout["text"],
        "ids_list": rollout["ids"].tolist() if hasattr(rollout["ids"], "tolist") else list(rollout["ids"]),
        "prompt_len": rollout.get("prompt_len", 0),
        "sentences": rollout.get("sentences", []),
        "token_ranges": rollout.get("token_ranges", []),
    }
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(data, f, indent=2)

def save_attention_data(attn_weights, save_dir, filename, top_heads):
    """Save attention data for specified heads only."""
    attn_dict = {}
    for layer, head in top_heads:
        if layer < len(attn_weights):
            layer_attn = attn_weights[layer]
            if head < layer_attn.shape[0]:
                key = f"L{layer}_H{head}"
                attn_dict[key] = layer_attn[head].numpy()
    
    print(f"     (Saving {len(attn_dict)} heads)")
    np.savez_compressed(os.path.join(save_dir, filename), **attn_dict)

# ================================
# Main logic
# ================================
def scan_existing_data(dataset):
    """Scan existing data and determine what needs to be populated."""
    base_dir = os.path.join(FINAL_DIR, dataset)
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return []
    
    results = []
    for folder in sorted(os.listdir(base_dir)):
        if folder == "aggregate":
            continue
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        config_path = os.path.join(folder_path, "config.json")
        if not os.path.exists(config_path):
            continue
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        pi = config.get("pi")
        faithful_indices = config.get("cued_professor_mention_indices", [])
        saved_cued_idx = config.get("cued_selected_rollout_idx")
        num_rollouts = config.get("num_rollouts_per_condition", 5)
        
        # Calculate unfaithful indices
        all_indices = set(range(num_rollouts))
        unfaithful_indices = list(all_indices - set(faithful_indices))
        
        # Check if FvU already exists
        fvu_dir = os.path.join(folder_path, "faithful_vs_unfaithful")
        has_faithful = os.path.exists(os.path.join(fvu_dir, "faithful", "rollout.json"))
        has_unfaithful = os.path.exists(os.path.join(fvu_dir, "unfaithful", "rollout.json"))
        has_fvu = has_faithful or has_unfaithful
        has_complete_fvu = has_faithful and has_unfaithful
        
        # Check config for consistently faithful/unfaithful flags
        consistently_faithful = config.get("consistently_faithful", False)
        consistently_unfaithful = config.get("consistently_unfaithful", False)
        
        # Can create FvU if we have at least one faithful OR one unfaithful rollout
        # (we'll try to generate the missing type, or mark as consistently faithful/unfaithful)
        can_create_fvu = len(faithful_indices) > 0 or len(unfaithful_indices) > 0
        
        # Check if saved cued rollout is faithful (can be reused)
        saved_is_faithful = saved_cued_idx in faithful_indices
        
        # Needs population if:
        # 1. No FvU at all, OR
        # 2. FvU is incomplete (missing faithful or unfaithful) AND not marked as consistently X
        needs_population = can_create_fvu and (
            not has_fvu or 
            (not has_complete_fvu and not consistently_faithful and not consistently_unfaithful)
        )
        
        results.append({
            "pi": pi,
            "folder": folder,
            "folder_path": folder_path,
            "faithful_indices": faithful_indices,
            "unfaithful_indices": unfaithful_indices,
            "saved_cued_idx": saved_cued_idx,
            "saved_is_faithful": saved_is_faithful,
            "has_fvu": has_fvu,
            "has_faithful": has_faithful,
            "has_unfaithful": has_unfaithful,
            "has_complete_fvu": has_complete_fvu,
            "consistently_faithful": consistently_faithful,
            "consistently_unfaithful": consistently_unfaithful,
            "can_create_fvu": can_create_fvu,
            "needs_population": needs_population,
        })
    
    return results

def populate_fvu_for_pi(info, problem, dataset):
    """Populate faithful_vs_unfaithful for a single PI."""
    print(f"\n{'='*60}")
    print(f"Processing PI {info['pi']} ({info['folder']})")
    print(f"{'='*60}")
    
    folder_path = info["folder_path"]
    max_new_tokens = MAX_NEW_TOKENS_GPQA if dataset == "gpqa" else MAX_NEW_TOKENS_MMLU
    prompt = problem.get("question_with_cue") or problem.get("question_cued")
    
    # Load top_heads for this PI
    with open(os.path.join(folder_path, "top_heads.json"), "r") as f:
        top_heads_data = json.load(f)
    
    # Combine cued and uncued heads
    cued_heads = [tuple(h) for h, _ in top_heads_data["cued"]]
    uncued_heads = [tuple(h) for h, _ in top_heads_data["uncued"]]
    all_heads = list(set(cued_heads + uncued_heads))
    
    # Create FvU directories
    fvu_dir = os.path.join(folder_path, "faithful_vs_unfaithful")
    faithful_dir = os.path.join(fvu_dir, "faithful")
    unfaithful_dir = os.path.join(fvu_dir, "unfaithful")
    os.makedirs(faithful_dir, exist_ok=True)
    os.makedirs(unfaithful_dir, exist_ok=True)
    
    # Determine what we have and what we need
    has_faithful = len(info["faithful_indices"]) > 0
    has_unfaithful = len(info["unfaithful_indices"]) > 0
    saved_is_unfaithful = info["saved_cued_idx"] in info["unfaithful_indices"]
    
    faithful_idx = None
    unfaithful_idx = None
    consistently_faithful = info.get("consistently_faithful", False)
    consistently_unfaithful = info.get("consistently_unfaithful", False)
    generation_faithful_rate = None
    
    max_attempts = 10  # 10 attempts for both directions
    
    # === FAITHFUL ROLLOUT ===
    if info.get("has_faithful"):
        # Already have faithful rollout in FvU folder
        print(f"  ✅ Faithful rollout already exists in FvU folder")
        faithful_idx = "existing"
    elif has_faithful and info["saved_is_faithful"]:
        # Reuse existing cued rollout (it's faithful)
        print(f"  ✅ Reusing existing faithful rollout from cued/")
        shutil.copy(
            os.path.join(folder_path, "cued", "rollout.json"),
            os.path.join(faithful_dir, "rollout.json")
        )
        shutil.copy(
            os.path.join(folder_path, "cued", "attention.npz"),
            os.path.join(faithful_dir, "attention.npz")
        )
        faithful_idx = info["saved_cued_idx"]
    else:
        # Try to get existing faithful rollout from CSV first (MMLU only)
        existing = get_existing_rollout_from_csv(info["pi"], faithful=True, dataset=dataset)
        
        if existing is not None:
            print(f"  📂 Found existing faithful rollout in CSV (response_idx={existing['response_idx']})")
            model, tokenizer = load_model()
            
            # Get the prompt and combine with model response
            model_response = existing['text']
            
            # Split prompt into sentences
            prompt_sentences = split_prompt_into_sentences(prompt)
            prompt_len = len(prompt_sentences)
            
            # Split model response into sentences
            response_sentences = split_into_sentences(model_response)
            
            # Combine: prompt + response
            all_sentences = prompt_sentences + response_sentences
            full_text = prompt + model_response
            
            # Tokenize the full text (prompt + response)
            ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
            
            faithful_rollout = {
                "ids": ids,
                "text": full_text,
                "prompt_len": prompt_len,
                "sentences": all_sentences
            }
            
            print(f"     Prompt: {prompt_len} sentences, Response: {len(response_sentences)} sentences")
            print(f"     Computing attention...")
            attn, orig_len = get_attention_weights(ids)
            if attn is None:
                print(f"     ❌ Could not compute attention (OOM), skipping faithful")
                consistently_unfaithful = True
                generation_faithful_rate = 0.0
            else:
                save_rollout_data(faithful_rollout, faithful_dir, "rollout.json")
                save_attention_data(attn, faithful_dir, "attention.npz", all_heads)
                del attn
                clear_cuda_memory()
                faithful_idx = existing['response_idx']
        else:
            # No CSV data - need to generate a new faithful rollout
            print(f"  🎲 Generating faithful rollout...")
            faithful_rollout = None
            unfaithful_count = 0
            oom_during_gen = False
            
            for attempt in range(max_attempts):
                rollout = generate_rollout(prompt, max_new_tokens, include_prompt=True)
                if rollout is None:
                    print(f"     ❌ OOM during generation on attempt {attempt + 1}")
                    oom_during_gen = True
                    break
                if mentions_cue(rollout["text"]):
                    print(f"     Got faithful rollout on attempt {attempt + 1}")
                    faithful_rollout = rollout
                    break
                else:
                    unfaithful_count += 1
                    print(f"     Attempt {attempt + 1}: unfaithful ({unfaithful_count}/{attempt + 1})")
            
            if oom_during_gen:
                # OOM during generation - skip this PI entirely
                print(f"     ⚠️ Skipping PI due to OOM during faithful generation")
                return False
            elif faithful_rollout is None:
                # Consistently unfaithful
                consistently_unfaithful = True
                generation_faithful_rate = 0.0
                print(f"     ⚠️ Consistently unfaithful after {max_attempts} attempts")
            else:
                # Save the faithful rollout - need to properly split prompt vs response
                # The rollout text includes prompt, so we need to extract just the response
                prompt_sentences = split_prompt_into_sentences(prompt)
                prompt_len = len(prompt_sentences)
                
                # The generated text includes the prompt - extract just the response part
                response_text = faithful_rollout["text"][len(prompt):]
                response_sentences = split_into_sentences(response_text)
                
                all_sentences = prompt_sentences + response_sentences
                faithful_rollout["sentences"] = all_sentences
                faithful_rollout["prompt_len"] = prompt_len
                
                print(f"     Prompt: {prompt_len} sentences, Response: {len(response_sentences)} sentences")
                print(f"     Computing attention...")
                attn, orig_len = get_attention_weights(faithful_rollout["ids"])
                if attn is None:
                    print(f"     ❌ Could not compute attention (OOM), skipping faithful")
                    consistently_unfaithful = True  # Mark as can't process
                    generation_faithful_rate = 0.0
                else:
                    save_rollout_data(faithful_rollout, faithful_dir, "rollout.json")
                    save_attention_data(attn, faithful_dir, "attention.npz", all_heads)
                    del attn
                    clear_cuda_memory()
                    faithful_idx = -1
    
    # === UNFAITHFUL ROLLOUT ===
    if info.get("has_unfaithful"):
        # Already have unfaithful rollout in FvU folder
        print(f"  ✅ Unfaithful rollout already exists in FvU folder")
        unfaithful_idx = "existing"
    elif has_unfaithful and saved_is_unfaithful:
        # Reuse existing cued rollout (it's unfaithful)
        print(f"  ✅ Reusing existing unfaithful rollout from cued/")
        shutil.copy(
            os.path.join(folder_path, "cued", "rollout.json"),
            os.path.join(unfaithful_dir, "rollout.json")
        )
        shutil.copy(
            os.path.join(folder_path, "cued", "attention.npz"),
            os.path.join(unfaithful_dir, "attention.npz")
        )
        unfaithful_idx = info["saved_cued_idx"]
    elif not consistently_unfaithful:
        # Try to get existing unfaithful rollout from CSV first (MMLU only)
        existing = get_existing_rollout_from_csv(info["pi"], faithful=False, dataset=dataset)
        
        if existing is not None:
            print(f"  📂 Found existing unfaithful rollout in CSV (response_idx={existing['response_idx']})")
            model, tokenizer = load_model()
            
            # Get the prompt and combine with model response
            model_response = existing['text']
            
            # Split prompt into sentences
            prompt_sentences = split_prompt_into_sentences(prompt)
            prompt_len = len(prompt_sentences)
            
            # Split model response into sentences
            response_sentences = split_into_sentences(model_response)
            
            # Combine: prompt + response
            all_sentences = prompt_sentences + response_sentences
            full_text = prompt + model_response
            
            # Tokenize the full text (prompt + response)
            ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
            
            unfaithful_rollout = {
                "ids": ids,
                "text": full_text,
                "prompt_len": prompt_len,
                "sentences": all_sentences
            }
            
            print(f"     Prompt: {prompt_len} sentences, Response: {len(response_sentences)} sentences")
            print(f"     Computing attention...")
            attn, orig_len = get_attention_weights(ids)
            if attn is None:
                print(f"     ❌ Could not compute attention (OOM), skipping unfaithful")
                consistently_faithful = True
                generation_faithful_rate = 1.0
            else:
                save_rollout_data(unfaithful_rollout, unfaithful_dir, "rollout.json")
                save_attention_data(attn, unfaithful_dir, "attention.npz", all_heads)
                del attn
                clear_cuda_memory()
                unfaithful_idx = existing['response_idx']
        else:
            # No CSV data - need to generate a new unfaithful rollout
            print(f"  🎲 Generating unfaithful rollout...")
            unfaithful_rollout = None
            faithful_count = 0
            oom_during_gen = False
            
            for attempt in range(max_attempts):
                rollout = generate_rollout(prompt, max_new_tokens, include_prompt=True)
                if rollout is None:
                    print(f"     ❌ OOM during generation on attempt {attempt + 1}")
                    oom_during_gen = True
                    break
                if not mentions_cue(rollout["text"]):
                    print(f"     Got unfaithful rollout on attempt {attempt + 1}")
                    unfaithful_rollout = rollout
                    break
                else:
                    faithful_count += 1
                    print(f"     Attempt {attempt + 1}: faithful ({faithful_count}/{attempt + 1})")
            
            if oom_during_gen:
                # OOM during generation - skip this PI entirely
                print(f"     ⚠️ Skipping PI due to OOM during unfaithful generation")
                return False
            elif unfaithful_rollout is None:
                # Consistently faithful
                consistently_faithful = True
                generation_faithful_rate = 1.0
                print(f"     ⚠️ Consistently faithful after {max_attempts} attempts")
            else:
                # Save the unfaithful rollout - need to properly split prompt vs response
                prompt_sentences = split_prompt_into_sentences(prompt)
                prompt_len = len(prompt_sentences)
                
                # The generated text includes the prompt - extract just the response part
                response_text = unfaithful_rollout["text"][len(prompt):]
                response_sentences = split_into_sentences(response_text)
                
                all_sentences = prompt_sentences + response_sentences
                unfaithful_rollout["sentences"] = all_sentences
                unfaithful_rollout["prompt_len"] = prompt_len
                
                print(f"     Prompt: {prompt_len} sentences, Response: {len(response_sentences)} sentences")
                print(f"     Computing attention...")
                attn, orig_len = get_attention_weights(unfaithful_rollout["ids"])
                if attn is None:
                    print(f"     ❌ Could not compute attention (OOM), skipping unfaithful")
                    consistently_faithful = True  # Mark as can't process
                    generation_faithful_rate = 1.0
                else:
                    save_rollout_data(unfaithful_rollout, unfaithful_dir, "rollout.json")
                    save_attention_data(attn, unfaithful_dir, "attention.npz", all_heads)
                    del attn
                    clear_cuda_memory()
                    unfaithful_idx = -1
    
    # === UPDATE CONFIG ===
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Clean up empty directories
    if consistently_faithful and os.path.exists(unfaithful_dir) and not os.listdir(unfaithful_dir):
        shutil.rmtree(unfaithful_dir)
    if consistently_unfaithful and os.path.exists(faithful_dir) and not os.listdir(faithful_dir):
        shutil.rmtree(faithful_dir)
    
    # Determine final status
    if consistently_faithful:
        print(f"  📝 Saving as consistently faithful (no unfaithful comparison available)")
        config["has_faithful_vs_unfaithful"] = True
        config["consistently_faithful"] = True
        config["consistently_unfaithful"] = False
        config["generation_faithful_rate"] = generation_faithful_rate
        config["faithful_cued_rollout_idx"] = faithful_idx
        config["unfaithful_cued_rollout_idx"] = None
    elif consistently_unfaithful:
        print(f"  📝 Saving as consistently unfaithful (no faithful comparison available)")
        config["has_faithful_vs_unfaithful"] = True
        config["consistently_faithful"] = False
        config["consistently_unfaithful"] = True
        config["generation_faithful_rate"] = generation_faithful_rate
        config["faithful_cued_rollout_idx"] = None
        config["unfaithful_cued_rollout_idx"] = unfaithful_idx
    else:
        # Have both!
        config["has_faithful_vs_unfaithful"] = True
        config["consistently_faithful"] = False
        config["consistently_unfaithful"] = False
        config["generation_faithful_rate"] = None
        config["faithful_cued_rollout_idx"] = faithful_idx
        config["unfaithful_cued_rollout_idx"] = unfaithful_idx
    
    config["cued_unfaithful_indices"] = info["unfaithful_indices"]
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✅ FvU data saved!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Populate faithful_vs_unfaithful data")
    parser.add_argument("--dataset", choices=["mmlu", "gpqa", "all"], default="all",
                        help="Which dataset to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be done, don't actually do it")
    parser.add_argument("--force-csv", action="store_true",
                        help="Force re-population using CSV data, ignoring consistently_faithful flags (MMLU only)")
    args = parser.parse_args()
    
    datasets = ["mmlu", "gpqa"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# Processing {dataset.upper()}")
        print(f"{'#'*60}")
        
        # Scan existing data
        results = scan_existing_data(dataset)
        
        # If --force-csv for MMLU, override the consistently_faithful flags for incomplete PIs
        if args.force_csv and dataset == "mmlu":
            for r in results:
                if not r["has_complete_fvu"]:
                    # Check if CSV has the missing type
                    if not r["has_unfaithful"]:
                        existing = get_existing_rollout_from_csv(r["pi"], faithful=False, dataset="mmlu")
                        if existing:
                            r["consistently_faithful"] = False
                            r["needs_population"] = True
                            print(f"  [force-csv] PI {r['pi']}: found unfaithful in CSV")
                    if not r["has_faithful"]:
                        existing = get_existing_rollout_from_csv(r["pi"], faithful=True, dataset="mmlu")
                        if existing:
                            r["consistently_unfaithful"] = False
                            r["needs_population"] = True
                            print(f"  [force-csv] PI {r['pi']}: found faithful in CSV")
        
        # Summary
        print(f"\nFound {len(results)} PIs in {dataset}:")
        needs_population = [r for r in results if r["needs_population"]]
        already_has = [r for r in results if r["has_fvu"]]
        cannot_create = [r for r in results if not r["can_create_fvu"]]
        
        print(f"  - Already has FvU: {len(already_has)} PIs")
        print(f"  - Needs population: {len(needs_population)} PIs")
        print(f"  - Cannot create (all faithful or all unfaithful): {len(cannot_create)} PIs")
        
        if needs_population:
            print(f"\nPIs to populate:")
            for r in needs_population:
                print(f"  - PI {r['pi']}: faithful={r['faithful_indices']}, unfaithful={r['unfaithful_indices']}")
                print(f"    Saved cued rollout {r['saved_cued_idx']} is {'faithful (reusable)' if r['saved_is_faithful'] else 'unfaithful'}")
        
        if args.dry_run:
            print("\n[DRY RUN] Would populate the above PIs")
            continue
        
        if not needs_population:
            print("\nNothing to populate!")
            continue
        
        # Load problems
        problems = load_problems(dataset)
        
        # Process each PI
        for info in needs_population:
            pi = info["pi"]
            if pi not in problems:
                print(f"\n⚠️ PI {pi} not found in problems file, skipping")
                continue
            
            problem = problems[pi]
            success = populate_fvu_for_pi(info, problem, dataset)
            
            # ALWAYS clear memory after each PI (success or not)
            clear_cuda_memory(verbose=True)
            
            if not success:
                print(f"  ⚠️ Failed to process PI {pi}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()

