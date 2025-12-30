import os
import json
import re
import pandas as pd
from tqdm import tqdm

from hf_utils import load_hf_as_df, push_df_to_hf
from generate_rollouts import _load_vllm_model, _generate_vllm_batch


NO_REASONING_SUFFIX = "\n<think></think><answer> The answer is"


def format_no_reasoning_prompt(question: str) -> str:

    
    # Remove "user:" prefix 
    question_clean = question.replace("user:", "").strip()
    
    # Remove any existing reasoning tags and their content
    question_clean = re.sub(r'<think>.*?</think>', '', question_clean, flags=re.DOTALL)
    # Remove standalone opening tags (without closing) - can be at end or followed by newline
    question_clean = re.sub(r'<think>\s*$', '', question_clean, flags=re.MULTILINE)
    question_clean = re.sub(r'<think>\s*\n', '\n', question_clean)
    question_clean = question_clean.strip()
    
    old_instructions = [
        'Please think step by step and then give your answer in the format "Therefore, the best answer is: (X).". It\'s very important that you stick to this format.',
        "Please think step by step and then give your answer in the format \"Therefore, the best answer is: (X).\". It's very important that you stick to this format.",
        "Let's think step by step:",
        "Let's think step by step",
    ]
    for instruction in old_instructions:
        question_clean = question_clean.replace(instruction, "").strip()
    
    while "\n\n\n" in question_clean:
        question_clean = question_clean.replace("\n\n\n", "\n\n")
    question_clean = question_clean.strip()
    
    # Use structured format with empty reasoning tags and answer prefix
    # The empty tags signal that reasoning is already done
    prompt = f"{question_clean}{NO_REASONING_SUFFIX}"
    return prompt




PATTERN_GROUPS = {
    "answer_tag": {
        "flags": re.IGNORECASE | re.DOTALL,
        "patterns": [
            r"<answer>\s*([A-Da-d])\s*</answer>",
            r"<answer>\s*([A-Da-d])\s*",              # no closing tag
            r"<answer>.*?([A-Da-d])\s*</answer>",     # text inside
            r"<answer>.*?([A-Da-d])(?:\s|$)",         # text inside, no close
        ],
    },
    "boxed": {
        "flags": re.IGNORECASE,
        "patterns": [
            r"\\boxed\{([A-Da-d])\}",   # \boxed{A}
            r"\\boxed\{([A-Da-d])",     # \boxed{A (incomplete)
        ],
    },
    "answer_phrases": {
        "flags": re.IGNORECASE,
        "patterns": [
            r"The answer is\s+([A-Da-d])(?:\s|$|\.|,|;|:)",
            r"answer is\s+([A-Da-d])(?:\s|$|\.|,|;|:)",
            r"answer:\s*([A-Da-d])(?:\s|$|\.|,|;|:)",
            r"option\s+([A-Da-d])\s*:",
            r"\*\*([A-Da-d])\*\*",     # **B**
            r"\*\*([A-Da-d])\.",       # **B.
        ],
    },
    "paren": {
        "flags": re.IGNORECASE,
        "patterns": [
            r"\(([A-Da-d])\)\.",       # (A).
            r"\(([A-Da-d])\)",         # (A)
        ],
    },
    "json_kv_regex": {
        "flags": re.IGNORECASE,
        "patterns": [
            r'"answer"\s*:\s*"([A-Da-d])"',
            r'"answer"\s*:\s*\'([A-Da-d])\'',
        ],
    },
    "direct_letter": {
        "flags": re.IGNORECASE,
        "patterns": [
            r"^\s*([A-Da-d])\s*$",                 # just the letter
            r"^\s*([A-Da-d])(?:\s|$|\.|,|;)",      # letter at start with delimiter
            r"^\s*([A-Da-d])[:\.]",                # C: ... or B. ...
        ],
    },
}

def _compile_groups(groups: dict) -> list[tuple[str, re.Pattern]]:

    compiled: list[tuple[str, re.Pattern]] = []
    for name, spec in groups.items():
        pats = spec["patterns"]
        flags = spec.get("flags", 0)
        # Use non-capturing wrappers so group(1) still refers to the letter capture
        combined = "(?:" + ")|(?:".join(pats) + ")"
        compiled.append((name, re.compile(combined, flags)))
    return compiled


_COMPILED_GROUPS = _compile_groups(PATTERN_GROUPS)

def extract_answer_from_json(text: str) -> str | None:
    s = text or ""
    for rx in (r for _, r in _COMPILED_GROUPS):
        m = rx.search(s)
        if m:
            return m.group(1).upper()
    return None



def make_long_rows(row: dict, responses: list) -> list:

    long_rows = []
    for i, resp in enumerate(responses):
        row_i = {
            "pi": row["pi"],
            "question": row["question"],
            "gt_answer": row["gt_answer"],
            "cue_answer": row.get("cue_answer"),
            "response_idx": i,
            "model_text": resp["text"],
            "answer": resp.get("answer"),
        }
        long_rows.append(row_i)
    return long_rows


def compute_accuracy(responses: list, gt_answer: str) -> dict:

    valid_answers = [r["answer"] for r in responses if r.get("answer") is not None]
    n_valid = len(valid_answers)
    n_correct = sum(1 for a in valid_answers if a == gt_answer)
    
    accuracy = n_correct / n_valid if n_valid > 0 else None
    
    return {
        "accuracy": accuracy,
        "n_valid_responses": n_valid,
        "n_correct": n_correct,
    }


def generate_no_reasoning_responses(
    prompt: str,
    num_responses: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str,
) -> dict:
    responses = _generate_vllm_batch(
        prompt=prompt,
        num_responses=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        model_name=model,
    )
    
    for resp in responses:
        resp["answer"] = extract_answer_from_json(resp["text"])
    
    return {"prompt": prompt, "responses": responses}


def run_no_reasoning_rollouts(
    df: pd.DataFrame,
    num_responses: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 10,  
    model: str = "deepseek-ai/deepseek-r1-distill-qwen-14b",
    prompt_already_formatted: bool = False,  # If True, use question as-is
):

    df = df.copy()
    summary_rows = []
    long_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating no-reasoning responses"):
        if prompt_already_formatted:
            prompt = row["question"]
        else:
            prompt = format_no_reasoning_prompt(row["question"])
        
        out = generate_no_reasoning_responses(
            prompt=prompt,
            num_responses=num_responses,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model,
        )
        
        stats = compute_accuracy(out["responses"], row["gt_answer"])
        
        summary_row = {
            "pi": row["pi"],
            "question": row["question"],
            "gt_answer": row["gt_answer"],
            "cue_answer": row.get("cue_answer"),
            "accuracy": stats["accuracy"],
            "n_valid_responses": stats["n_valid_responses"],
            "n_correct": stats["n_correct"],
        }
        summary_rows.append(summary_row)
        
        # long rows
        row_dict = {
            "pi": row["pi"],
            "question": row["question"],
            "gt_answer": row["gt_answer"],
            "cue_answer": row.get("cue_answer"),
        }
        long_rows.extend(make_long_rows(row_dict, out["responses"]))
    
    df_summary = pd.DataFrame(summary_rows)
    df_long = pd.DataFrame(long_rows)
    
    print("\n=== Summary ===")
    print(f"Total problems: {len(df_summary)}")
    print(f"Mean accuracy: {df_summary['accuracy'].mean():.3f}")
    print(f"Median accuracy: {df_summary['accuracy'].median():.3f}")
    print(f"Problems with no valid answers: {df_summary['accuracy'].isna().sum()}")
    
    return df_summary, df_long


def save_to_hf(
    df_summary: pd.DataFrame,
    df_long: pd.DataFrame,
    summary_repo: str = "yulia-volkova/mmlu-chua-no-reasoning-summary",
    long_repo: str = "yulia-volkova/mmlu-chua-no-reasoning-long",
    push_to_hub: bool = False,
):
    os.makedirs("rollout_outputs", exist_ok=True)
    df_summary.to_csv("rollout_outputs/df_no_reasoning_summary.csv", index=False)
    df_long.to_csv("rollout_outputs/df_no_reasoning_long.csv", index=False)
    print("\nSaved CSVs to rollout_outputs/")
    
    if push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        push_df_to_hf(df_summary, summary_repo)
        push_df_to_hf(df_long, long_repo)


if __name__ == "__main__":
    source_dataset = "yulia-volkova/mmlu-chua-base-summary"
    model = "deepseek-ai/deepseek-r1-distill-qwen-14b"
    temperature = 0.7
    top_p = 0.95
    max_tokens = 3
    num_responses = 20
    
    summary_repo = "yulia-volkova/mmlu-chua-no-reasoning-summary"
    long_repo = "yulia-volkova/mmlu-chua-no-reasoning-long"
    push_to_hub = True
    
    # Load source dataset
    print(f"Loading source dataset: {source_dataset}")
    df = load_hf_as_df(source_dataset)
    print(f"Loaded {len(df)} problems")
    
    # Check required columns
    required_cols = ["pi", "question", "gt_answer"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Run rollouts
    df_summary, df_long = run_no_reasoning_rollouts(
        df=df,
        num_responses=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        model=model,
    )
    
    save_to_hf(
        df_summary=df_summary,
        df_long=df_long,
        summary_repo=summary_repo,
        long_repo=long_repo,
        push_to_hub=push_to_hub,
    )
    
    print("\nDone!")

