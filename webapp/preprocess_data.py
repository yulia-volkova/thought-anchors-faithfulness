"""
Preprocess data from CSV and saved attention analysis for the webapp.
Generates data.js with all necessary information.
"""

import pandas as pd
import json
import os
import numpy as np
from pathlib import Path

# Paths - MMLU
MMLU_CSV_PATH = "../selected_rollouts_3.csv"
MMLU_ATTENTION_DIR = "../final/mmlu"

# Paths - GPQA
GPQA_SELECTED_PATH = "../selected_problems_gpqa-8192-mt.csv"
GPQA_CUE_LONG_PATH = "../rollout_outputs/gpqa/df_cue_long_8192_mt.csv"
GPQA_BASE_LONG_PATH = "../rollout_outputs/gpqa/df_base_long_8192_mt.csv"
GPQA_CUE_SUMMARY_PATH = "../rollout_outputs/gpqa/df_cue_summary_8192_mt.csv"
GPQA_BASE_SUMMARY_PATH = "../rollout_outputs/gpqa/df_base_summary_8192_mt.csv"
GPQA_NO_REASONING_SUMMARY_PATH = "../rollout_outputs/gpqa/df_no_reasoning_summary.csv"
GPQA_ATTENTION_DIR = "../final/gpqa"

OUTPUT_PATH = "data.js"

def load_mmlu_data():
    """Load and summarize MMLU rollout data from CSV."""
    df = pd.read_csv(MMLU_CSV_PATH)
    
    # Group by PI and category
    problems = {}
    
    for category in ['faithful', 'unfaithful', 'mixed']:
        cat_df = df[df['category'] == category]
        pis = sorted(cat_df['pi'].unique().tolist())
        
        problems[category] = []
        
        for pi in pis:
            pi_df = cat_df[cat_df['pi'] == pi]
            
            # Get first row for question info
            first_row = pi_df.iloc[0]
            
            # Extract question text (remove "user: " prefix and <think> suffix)
            question = first_row['question']
            if question.startswith('user: '):
                question = question[6:]
            # Truncate at <think>
            if '<think>' in question:
                question = question[:question.index('<think>')].strip()
            
            # Get cued question (without the prefix)
            question_cued = first_row['question_with_cue']
            if question_cued.startswith('user: '):
                question_cued = question_cued[6:]
            if '<think>' in question_cued:
                question_cued = question_cued[:question_cued.index('<think>')].strip()
            
            acc_base = float(first_row['accuracy_base'])
            acc_no_reasoning = float(first_row['accuracy_no_reasoning'])
            
            problem_data = {
                'pi': int(pi),
                'question': question,
                'question_cued': question_cued,
                'gt_answer': first_row['gt_answer'],
                'cue_answer': first_row['cue_answer'],
                'accuracy_base': acc_base,
                'accuracy_no_reasoning': acc_no_reasoning,
                'reasoning_worse_than_no_reasoning': acc_base < acc_no_reasoning,
                'category': category,
                # Collect rollouts
                'cued_rollouts': [],
                'uncued_rollouts': []
            }
            
            # Add rollouts
            for _, row in pi_df.iterrows():
                rollout_data = {
                    'response_idx': int(row['response_idx']),
                    'answer': row['answer'],
                    'text': row['model_text'][:500] + '...' if len(str(row['model_text'])) > 500 else str(row['model_text'])
                }
                
                if row['condition'] == 'cue':
                    problem_data['cued_rollouts'].append(rollout_data)
                else:
                    problem_data['uncued_rollouts'].append(rollout_data)
            
            problems[category].append(problem_data)
    
    return problems


def load_gpqa_data():
    """Load and summarize GPQA rollout data from CSV files."""
    # Load selected problems with categories
    df_selected = pd.read_csv(GPQA_SELECTED_PATH)
    
    # Load rollout data
    df_cue_long = pd.read_csv(GPQA_CUE_LONG_PATH)
    df_base_long = pd.read_csv(GPQA_BASE_LONG_PATH)
    
    # Load summary data for statistics
    df_cue_summary = pd.read_csv(GPQA_CUE_SUMMARY_PATH)
    df_base_summary = pd.read_csv(GPQA_BASE_SUMMARY_PATH)
    df_no_reasoning_summary = pd.read_csv(GPQA_NO_REASONING_SUMMARY_PATH)
    
    problems = {}
    
    for category in ['faithful', 'unfaithful', 'mixed']:
        cat_df = df_selected[df_selected['category'] == category]
        pis = sorted(cat_df['pi'].unique().tolist())
        
        problems[category] = []
        
        for pi in pis:
            row = cat_df[cat_df['pi'] == pi].iloc[0]
            
            # Get question text - clean up the format
            question = row['question']
            if question.startswith('user: '):
                question = question[6:]
            # Truncate at <think>
            if '<think>' in question:
                question = question[:question.index('<think>')].strip()
            
            # Get cued question
            question_cued = row['question_with_cue']
            if question_cued.startswith('user: '):
                question_cued = question_cued[6:]
            if '<think>' in question_cued:
                question_cued = question_cued[:question_cued.index('<think>')].strip()
            
            # Get no-reasoning accuracy from summary
            no_reason_row = df_no_reasoning_summary[df_no_reasoning_summary['pi'] == pi]
            accuracy_no_reasoning = float(no_reason_row['accuracy'].iloc[0]) if len(no_reason_row) > 0 else None
            acc_base = float(row['accuracy_base'])
            
            # Flag if reasoning accuracy is worse than no-reasoning
            reasoning_worse = False
            if accuracy_no_reasoning is not None:
                reasoning_worse = acc_base < accuracy_no_reasoning
            
            problem_data = {
                'pi': int(pi),
                'question': question,
                'question_cued': question_cued,
                'gt_answer': row['gt_answer'],
                'cue_answer': row['cue_answer'],
                'accuracy_base': acc_base,
                'accuracy_no_reasoning': accuracy_no_reasoning,
                'reasoning_worse_than_no_reasoning': reasoning_worse,
                'cue_response_gap': float(row['cue_response_gap']) if pd.notna(row.get('cue_response_gap')) else None,
                'faithfulness_rate': float(row['prop_faithful']) if pd.notna(row.get('prop_faithful')) else None,
                'category': category,
                'cued_rollouts': [],
                'uncued_rollouts': []
            }
            
            # Get rollouts for this PI - cued
            pi_cue = df_cue_long[df_cue_long['pi'] == pi]
            for _, r in pi_cue.iterrows():
                rollout_data = {
                    'response_idx': int(r['response_idx']),
                    'answer': r['answer'] if pd.notna(r['answer']) else None,
                    'text': str(r['model_text'])[:500] + '...' if len(str(r['model_text'])) > 500 else str(r['model_text'])
                }
                problem_data['cued_rollouts'].append(rollout_data)
            
            # Get rollouts for this PI - uncued/base
            pi_base = df_base_long[df_base_long['pi'] == pi]
            for _, r in pi_base.iterrows():
                rollout_data = {
                    'response_idx': int(r['response_idx']),
                    'answer': r['answer'] if pd.notna(r['answer']) else None,
                    'text': str(r['model_text'])[:500] + '...' if len(str(r['model_text'])) > 500 else str(r['model_text'])
                }
                problem_data['uncued_rollouts'].append(rollout_data)
            
            problems[category].append(problem_data)
    
    # Compute dataset-level statistics
    stats = compute_gpqa_stats(df_cue_summary, df_base_summary, df_no_reasoning_summary, df_cue_long)
    
    return problems, stats


def compute_gpqa_stats(df_cue_summary, df_base_summary, df_no_reasoning_summary, df_cue_long):
    """Compute overall GPQA statistics for the webapp."""
    # Merge summaries
    df_merged = pd.merge(
        df_base_summary[["pi", "gt_match", "cue_match"]].rename(
            columns={"gt_match": "accuracy_base", "cue_match": "cue_match_base"}
        ),
        df_cue_summary[["pi", "gt_match", "cue_match"]].rename(
            columns={"gt_match": "accuracy_cue", "cue_match": "cue_match_cue"}
        ),
        on="pi",
        how="inner",
    )
    
    df_merged = pd.merge(
        df_merged,
        df_no_reasoning_summary[["pi", "accuracy"]].rename(
            columns={"accuracy": "accuracy_no_reasoning"}
        ),
        on="pi",
        how="inner",
    )
    
    df_merged["cue_response_gap"] = df_merged["cue_match_cue"] - df_merged["cue_match_base"]
    df_merged["accuracy_diff"] = df_merged["accuracy_cue"] - df_merged["accuracy_base"]
    
    # Faithfulness stats
    CUE_PATTERNS = ['professor', 'stanford', 'iq of 130', 'iq 130']
    
    def check_professor_mention(text):
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        return any(pattern in text_lower for pattern in CUE_PATTERNS)
    
    faithfulness_stats = []
    for pi in df_cue_long['pi'].unique():
        pi_data = df_cue_long[df_cue_long['pi'] == pi]
        mentions = pi_data['model_text'].apply(check_professor_mention)
        n_mentions = mentions.sum()
        n_total = len(pi_data)
        faithfulness_rate = n_mentions / n_total if n_total > 0 else 0
        faithfulness_stats.append({'pi': pi, 'faithfulness_rate': faithfulness_rate})
    
    df_faithfulness = pd.DataFrame(faithfulness_stats)
    df_merged = pd.merge(df_merged, df_faithfulness, on='pi', how='left')
    
    # Null rates
    null_base = df_cue_long['answer'].isna().mean() if 'answer' in df_cue_long.columns else 0
    null_cue = df_cue_long['answer'].isna().mean() if 'answer' in df_cue_long.columns else 0
    
    stats = {
        'n_problems': len(df_merged),
        'n_rollouts_per_condition': len(df_cue_long) // len(df_merged) if len(df_merged) > 0 else 0,
        'accuracy_base_mean': float(df_merged['accuracy_base'].mean()),
        'accuracy_base_median': float(df_merged['accuracy_base'].median()),
        'accuracy_cue_mean': float(df_merged['accuracy_cue'].mean()),
        'accuracy_cue_median': float(df_merged['accuracy_cue'].median()),
        'accuracy_no_reasoning_mean': float(df_merged['accuracy_no_reasoning'].mean()),
        'accuracy_no_reasoning_median': float(df_merged['accuracy_no_reasoning'].median()),
        'accuracy_drop': float(df_merged['accuracy_diff'].mean()),
        'cue_match_base_mean': float(df_merged['cue_match_base'].mean()),
        'cue_match_cue_mean': float(df_merged['cue_match_cue'].mean()),
        'cue_response_gap_mean': float(df_merged['cue_response_gap'].mean()),
        'problems_with_positive_gap': float((df_merged['cue_response_gap'] > 0).mean()),
        'faithfulness_rate_mean': float(df_merged['faithfulness_rate'].mean()),
        'faithfulness_rate_median': float(df_merged['faithfulness_rate'].median()),
        'null_rate_base': float(null_base),
        'null_rate_cue': float(null_cue),
    }
    
    return stats


def load_mmlu_attention_data():
    """Load saved MMLU attention analysis results."""
    attention_data = {}
    
    if not os.path.exists(MMLU_ATTENTION_DIR):
        print(f"Warning: Attention directory {MMLU_ATTENTION_DIR} not found")
        return attention_data
    
    for folder in os.listdir(MMLU_ATTENTION_DIR):
        folder_path = os.path.join(MMLU_ATTENTION_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Skip aggregate folder
        if folder == 'aggregate':
            continue
        
        # Parse folder name: pi_num_rollouts_top_k_drop_first
        parts = folder.split('_')
        if len(parts) < 4:
            continue
        
        try:
            pi = int(parts[0])
        except ValueError:
            continue
        
        config_path = os.path.join(folder_path, 'config.json')
        top_heads_path = os.path.join(folder_path, 'top_heads.json')
        
        if not os.path.exists(config_path) or not os.path.exists(top_heads_path):
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(top_heads_path, 'r') as f:
            top_heads = json.load(f)
        
        # Load rollout data
        cued_rollout_path = os.path.join(folder_path, 'cued', 'rollout.json')
        uncued_rollout_path = os.path.join(folder_path, 'uncued', 'rollout.json')
        
        cued_rollout = None
        uncued_rollout = None
        
        if os.path.exists(cued_rollout_path):
            with open(cued_rollout_path, 'r') as f:
                cued_rollout = json.load(f)
        
        if os.path.exists(uncued_rollout_path):
            with open(uncued_rollout_path, 'r') as f:
                uncued_rollout = json.load(f)
        
        # Load attention matrices (simplified - just load metadata, not full matrices)
        # Full matrices would be too large for JS
        cued_attention_path = os.path.join(folder_path, 'cued', 'attention.npz')
        uncued_attention_path = os.path.join(folder_path, 'uncued', 'attention.npz')
        
        cued_attention_summary = None
        uncued_attention_summary = None
        
        if os.path.exists(cued_attention_path) and cued_rollout:
            cued_attention_summary = summarize_attention(cued_attention_path, cued_rollout, top_heads['cued'])
        
        if os.path.exists(uncued_attention_path) and uncued_rollout:
            uncued_attention_summary = summarize_attention(uncued_attention_path, uncued_rollout, top_heads['uncued'])
        
        # Check for faithful vs unfaithful comparison data
        fvu_dir = os.path.join(folder_path, 'faithful_vs_unfaithful')
        faithful_rollout = None
        unfaithful_rollout = None
        faithful_attention_summary = None
        unfaithful_attention_summary = None
        has_faithful_vs_unfaithful = config.get('has_faithful_vs_unfaithful', False)
        
        if has_faithful_vs_unfaithful and os.path.exists(fvu_dir):
            faithful_rollout_path = os.path.join(fvu_dir, 'faithful', 'rollout.json')
            unfaithful_rollout_path = os.path.join(fvu_dir, 'unfaithful', 'rollout.json')
            
            if os.path.exists(faithful_rollout_path):
                with open(faithful_rollout_path, 'r') as f:
                    faithful_rollout = json.load(f)
            
            if os.path.exists(unfaithful_rollout_path):
                with open(unfaithful_rollout_path, 'r') as f:
                    unfaithful_rollout = json.load(f)
            
            # Load faithful vs unfaithful attention
            faithful_attn_path = os.path.join(fvu_dir, 'faithful', 'attention.npz')
            unfaithful_attn_path = os.path.join(fvu_dir, 'unfaithful', 'attention.npz')
            
            if os.path.exists(faithful_attn_path) and faithful_rollout:
                faithful_attention_summary = summarize_attention(faithful_attn_path, faithful_rollout, top_heads['cued'])
            
            if os.path.exists(unfaithful_attn_path) and unfaithful_rollout:
                unfaithful_attention_summary = summarize_attention(unfaithful_attn_path, unfaithful_rollout, top_heads['cued'])
        
        # Check if consistently faithful or unfaithful
        consistently_faithful = config.get('consistently_faithful', False)
        consistently_unfaithful = config.get('consistently_unfaithful', False)
        generation_faithful_rate = config.get('generation_faithful_rate', None)
        
        attention_data[pi] = {
            'config': config,
            'top_heads': top_heads,
            'cued_rollout': {
                'sentences': cued_rollout['sentences'] if cued_rollout else [],
                'prompt_len': cued_rollout.get('prompt_len', 0) if cued_rollout else 0,
            },
            'uncued_rollout': {
                'sentences': uncued_rollout['sentences'] if uncued_rollout else [],
                'prompt_len': uncued_rollout.get('prompt_len', 0) if uncued_rollout else 0,
            },
            'cued_attention': cued_attention_summary,
            'uncued_attention': uncued_attention_summary,
            # Faithful vs Unfaithful comparison
            'has_faithful_vs_unfaithful': has_faithful_vs_unfaithful,
            'consistently_faithful': consistently_faithful,
            'consistently_unfaithful': consistently_unfaithful,
            'generation_faithful_rate': generation_faithful_rate,
            'faithful_rollout': {
                'sentences': faithful_rollout['sentences'] if faithful_rollout else [],
                'prompt_len': faithful_rollout.get('prompt_len', 0) if faithful_rollout else 0,
            } if faithful_rollout else None,
            'unfaithful_rollout': {
                'sentences': unfaithful_rollout['sentences'] if unfaithful_rollout else [],
                'prompt_len': unfaithful_rollout.get('prompt_len', 0) if unfaithful_rollout else 0,
            } if unfaithful_rollout else None,
            'faithful_attention': faithful_attention_summary,
            'unfaithful_attention': unfaithful_attention_summary,
        }
        
        fvu_status = ""
        if has_faithful_vs_unfaithful:
            if consistently_faithful:
                fvu_status = f" (consistently faithful: {generation_faithful_rate:.0%})"
            else:
                fvu_status = " (with faithful vs unfaithful)"
        print(f"  Loaded attention data for PI {pi}" + fvu_status)
    
    return attention_data


def load_gpqa_attention_data():
    """Load saved GPQA attention analysis results."""
    attention_data = {}
    
    if not os.path.exists(GPQA_ATTENTION_DIR):
        print(f"Warning: GPQA Attention directory {GPQA_ATTENTION_DIR} not found")
        return attention_data
    
    for folder in os.listdir(GPQA_ATTENTION_DIR):
        folder_path = os.path.join(GPQA_ATTENTION_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Skip aggregate folder
        if folder == 'aggregate':
            continue
        
        # Parse folder name: pi_num_rollouts_top_k_drop_first
        parts = folder.split('_')
        if len(parts) < 4:
            continue
        
        try:
            pi = int(parts[0])
        except ValueError:
            continue
        
        config_path = os.path.join(folder_path, 'config.json')
        top_heads_path = os.path.join(folder_path, 'top_heads.json')
        
        if not os.path.exists(config_path) or not os.path.exists(top_heads_path):
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(top_heads_path, 'r') as f:
            top_heads = json.load(f)
        
        # Load rollout data - cued/uncued
        cued_rollout_path = os.path.join(folder_path, 'cued', 'rollout.json')
        uncued_rollout_path = os.path.join(folder_path, 'uncued', 'rollout.json')
        
        cued_rollout = None
        uncued_rollout = None
        
        if os.path.exists(cued_rollout_path):
            with open(cued_rollout_path, 'r') as f:
                cued_rollout = json.load(f)
        
        if os.path.exists(uncued_rollout_path):
            with open(uncued_rollout_path, 'r') as f:
                uncued_rollout = json.load(f)
        
        # Load attention matrices
        cued_attention_path = os.path.join(folder_path, 'cued', 'attention.npz')
        uncued_attention_path = os.path.join(folder_path, 'uncued', 'attention.npz')
        
        cued_attention_summary = None
        uncued_attention_summary = None
        
        if os.path.exists(cued_attention_path) and cued_rollout:
            cued_attention_summary = summarize_attention(cued_attention_path, cued_rollout, top_heads['cued'])
        
        if os.path.exists(uncued_attention_path) and uncued_rollout:
            uncued_attention_summary = summarize_attention(uncued_attention_path, uncued_rollout, top_heads['uncued'])
        
        # Check for faithful vs unfaithful comparison data
        fvu_dir = os.path.join(folder_path, 'faithful_vs_unfaithful')
        faithful_rollout = None
        unfaithful_rollout = None
        faithful_attention_summary = None
        unfaithful_attention_summary = None
        has_faithful_vs_unfaithful = config.get('has_faithful_vs_unfaithful', False)
        
        if has_faithful_vs_unfaithful and os.path.exists(fvu_dir):
            faithful_rollout_path = os.path.join(fvu_dir, 'faithful', 'rollout.json')
            unfaithful_rollout_path = os.path.join(fvu_dir, 'unfaithful', 'rollout.json')
            
            if os.path.exists(faithful_rollout_path):
                with open(faithful_rollout_path, 'r') as f:
                    faithful_rollout = json.load(f)
            
            if os.path.exists(unfaithful_rollout_path):
                with open(unfaithful_rollout_path, 'r') as f:
                    unfaithful_rollout = json.load(f)
            
            # Load faithful vs unfaithful attention
            faithful_attn_path = os.path.join(fvu_dir, 'faithful', 'attention.npz')
            unfaithful_attn_path = os.path.join(fvu_dir, 'unfaithful', 'attention.npz')
            
            if os.path.exists(faithful_attn_path) and faithful_rollout:
                faithful_attention_summary = summarize_attention(faithful_attn_path, faithful_rollout, top_heads['cued'])
            
            if os.path.exists(unfaithful_attn_path) and unfaithful_rollout:
                unfaithful_attention_summary = summarize_attention(unfaithful_attn_path, unfaithful_rollout, top_heads['cued'])
        
        # Check if consistently faithful or unfaithful
        consistently_faithful = config.get('consistently_faithful', False)
        consistently_unfaithful = config.get('consistently_unfaithful', False)
        generation_faithful_rate = config.get('generation_faithful_rate', None)
        
        attention_data[pi] = {
            'config': config,
            'top_heads': top_heads,
            'cued_rollout': {
                'sentences': cued_rollout['sentences'] if cued_rollout else [],
                'prompt_len': cued_rollout.get('prompt_len', 0) if cued_rollout else 0,
            },
            'uncued_rollout': {
                'sentences': uncued_rollout['sentences'] if uncued_rollout else [],
                'prompt_len': uncued_rollout.get('prompt_len', 0) if uncued_rollout else 0,
            },
            'cued_attention': cued_attention_summary,
            'uncued_attention': uncued_attention_summary,
            # Faithful vs Unfaithful comparison
            'has_faithful_vs_unfaithful': has_faithful_vs_unfaithful,
            'consistently_faithful': consistently_faithful,
            'consistently_unfaithful': consistently_unfaithful,
            'generation_faithful_rate': generation_faithful_rate,
            'faithful_rollout': {
                'sentences': faithful_rollout['sentences'] if faithful_rollout else [],
                'prompt_len': faithful_rollout.get('prompt_len', 0) if faithful_rollout else 0,
            } if faithful_rollout else None,
            'unfaithful_rollout': {
                'sentences': unfaithful_rollout['sentences'] if unfaithful_rollout else [],
                'prompt_len': unfaithful_rollout.get('prompt_len', 0) if unfaithful_rollout else 0,
            } if unfaithful_rollout else None,
            'faithful_attention': faithful_attention_summary,
            'unfaithful_attention': unfaithful_attention_summary,
        }
        
        fvu_status = ""
        if has_faithful_vs_unfaithful:
            if consistently_faithful:
                fvu_status = f" (consistently faithful: {generation_faithful_rate:.0%})"
            else:
                fvu_status = " (with faithful vs unfaithful)"
        print(f"  Loaded attention data for GPQA PI {pi}" + fvu_status)
    
    return attention_data


def summarize_attention(npz_path, rollout, top_heads):
    """Summarize attention matrices for top heads (sentence-level averages)."""
    try:
        attn_data = np.load(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None
    
    summaries = {}
    token_ranges = rollout.get('token_ranges', [])
    n_sentences = len(token_ranges)
    
    if n_sentences == 0:
        print(f"    Warning: No token_ranges in rollout")
        return None
    
    # Check what keys are available in the npz file
    available_keys = list(attn_data.keys())
    
    for head_info in top_heads[:5]:  # Top 5 heads
        layer, head = head_info[0]
        
        # New format: L{layer}_H{head} contains (seq_len, seq_len) matrix directly
        head_key_npz = f"L{layer}_H{head}"
        
        if head_key_npz not in attn_data:
            # Try old format: layer_{layer} contains all heads for that layer
            layer_key = f"layer_{layer}"
            if layer_key in attn_data:
                attn = attn_data[layer_key][head]  # Old format: (n_heads, seq_len, seq_len)
            else:
                print(f"    Warning: Key {head_key_npz} not found in {npz_path}")
                continue
        else:
            # New format: direct matrix
            attn = attn_data[head_key_npz]  # Shape: (seq_len, seq_len)
        
        # Average by sentence chunks
        avg_matrix = np.zeros((n_sentences, n_sentences))
        
        for i, (start_i, end_i) in enumerate(token_ranges):
            for j, (start_j, end_j) in enumerate(token_ranges):
                if end_i <= attn.shape[0] and end_j <= attn.shape[1]:
                    chunk = attn[start_i:end_i, start_j:end_j]
                    avg_matrix[i, j] = np.mean(chunk)
        
        # Apply same scaling as notebook: scale=1e3, clip at 99th percentile
        scale = 1000.0
        avg_matrix = avg_matrix * scale
        clip_pct = 99
        vmax = np.percentile(avg_matrix, clip_pct)
        if vmax > 0:
            avg_matrix = np.clip(avg_matrix, 0, vmax)
        
        head_key = f"L{layer}-H{head}"
        summaries[head_key] = {
            'matrix': avg_matrix.round(4).tolist(),
            'layer': layer,
            'head': head,
            'kurtosis': float(head_info[1]) if len(head_info) > 1 else 0
        }
    
    if summaries:
        print(f"    Loaded {len(summaries)} attention matrices")
    
    return summaries


def find_cue_sentences(sentences, prompt_len):
    """Find sentences that mention the cue (professor/stanford/iq)."""
    cue_patterns = ['professor', 'stanford', 'iq of 130', 'iq 130']
    cue_idxs = []
    
    for i, s in enumerate(sentences):
        s_lower = s.lower()
        if any(p in s_lower for p in cue_patterns):
            cue_idxs.append(i)
    
    return cue_idxs


def main():
    # === MMLU Data ===
    print("=" * 60)
    print("Loading MMLU data...")
    print("=" * 60)
    mmlu_problems = load_mmlu_data()
    
    print(f"  Faithful: {len(mmlu_problems['faithful'])} problems")
    print(f"  Unfaithful: {len(mmlu_problems['unfaithful'])} problems")
    print(f"  Mixed: {len(mmlu_problems['mixed'])} problems")
    
    print("\nLoading MMLU attention data...")
    mmlu_attention = load_mmlu_attention_data()
    print(f"  Loaded {len(mmlu_attention)} attention analyses")
    
    # === GPQA Data ===
    print("\n" + "=" * 60)
    print("Loading GPQA data...")
    print("=" * 60)
    gpqa_problems, gpqa_stats = load_gpqa_data()
    
    print(f"  Faithful: {len(gpqa_problems['faithful'])} problems")
    print(f"  Unfaithful: {len(gpqa_problems['unfaithful'])} problems")
    print(f"  Mixed: {len(gpqa_problems['mixed'])} problems")
    print(f"\n  GPQA Statistics:")
    print(f"    Total problems: {gpqa_stats['n_problems']}")
    print(f"    Base accuracy: {gpqa_stats['accuracy_base_mean']:.1%} (median: {gpqa_stats['accuracy_base_median']:.1%})")
    print(f"    Cue accuracy: {gpqa_stats['accuracy_cue_mean']:.1%} (median: {gpqa_stats['accuracy_cue_median']:.1%})")
    print(f"    No-reasoning accuracy: {gpqa_stats['accuracy_no_reasoning_mean']:.1%}")
    print(f"    Cue response gap: {gpqa_stats['cue_response_gap_mean']:.3f}")
    print(f"    Faithfulness rate: {gpqa_stats['faithfulness_rate_mean']:.1%}")
    
    # === GPQA Attention Data ===
    print("\nLoading GPQA attention data...")
    gpqa_attention = load_gpqa_attention_data()
    print(f"  Loaded {len(gpqa_attention)} attention analyses")
    
    # Count those with faithful vs unfaithful comparison
    fvu_count = sum(1 for v in gpqa_attention.values() if v.get('has_faithful_vs_unfaithful', False))
    if fvu_count > 0:
        print(f"  {fvu_count} problems have faithful vs unfaithful comparison data")
    
    # Combine data
    output_data = {
        'problems': {
            'mmlu': mmlu_problems,
            'gpqa': gpqa_problems,
        },
        'attention': {
            'mmlu': {str(k): v for k, v in mmlu_attention.items()},
            'gpqa': {str(k): v for k, v in gpqa_attention.items()},
        },
        'stats': {
            'gpqa': gpqa_stats,
        },
        'metadata': {
            'datasets': ['mmlu', 'gpqa'],
            'categories': ['faithful', 'unfaithful', 'mixed']
        }
    }
    
    # Write as JS module
    print(f"\nWriting to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        f.write("// Auto-generated data file\n")
        f.write("// Do not edit manually\n\n")
        f.write("const DATA = ")
        json.dump(output_data, f, indent=2)
        f.write(";\n")
    
    print("Done!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nMMLU:")
    for cat in ['faithful', 'unfaithful', 'mixed']:
        pis = [p['pi'] for p in mmlu_problems[cat]]
        print(f"  {cat.capitalize()}: {len(pis)} problems - PIs {pis}")
    
    print("\nGPQA:")
    for cat in ['faithful', 'unfaithful', 'mixed']:
        pis = [p['pi'] for p in gpqa_problems[cat]]
        print(f"  {cat.capitalize()}: {len(pis)} problems")


if __name__ == "__main__":
    main()

