"""
Verify that GPQA questions and answers match between two HuggingFace sources:
1. Idavidrein/gpqa (original dataset from paper authors)
2. fingertap/GPQA-Diamond (reformatted with MCQ embedded)

Checks:
- Question text matches
- Correct answer matches
- Wrong answer options match

Usage:
    python verify_gpqa_sources.py

    # Or with HF token if needed:
    HF_TOKEN=your_token python verify_gpqa_sources.py
"""

import os
import re
from collections import defaultdict
from datasets import load_dataset
import pandas as pd


def load_idavidrein_gpqa():
    """Load original GPQA dataset from Idavidrein."""
    print("Loading Idavidrein/gpqa...")
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", token=token)
    df = ds["train"].to_pandas()
    print(f"  Loaded {len(df)} examples")
    return df


def load_fingertap_gpqa():
    """Load fingertap GPQA-Diamond dataset."""
    print("Loading fingertap/GPQA-Diamond...")
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("fingertap/GPQA-Diamond", token=token)
    df = ds["test"].to_pandas()
    print(f"  Loaded {len(df)} examples")
    return df


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


def extract_question_stem(fingertap_question: str) -> str:
    """Extract question stem from fingertap format (before answer choices)."""
    # fingertap format: "Question text... A. answer1 B. answer2 C. answer3 D. answer4"
    # Try to find where answer choices start
    patterns = [
        r'\s*A\.\s',  # "A. "
        r'\s*\(A\)\s',  # "(A) "
        r'\s*A\)\s',  # "A) "
    ]

    for pattern in patterns:
        match = re.search(pattern, fingertap_question)
        if match:
            return fingertap_question[:match.start()].strip()

    # If no pattern found, return first 500 chars as fallback
    return fingertap_question[:500].strip()


def extract_answer_choices(fingertap_question: str) -> dict:
    """Extract answer choices A, B, C, D from fingertap question."""
    choices = {}

    # Try different patterns
    # Pattern 1: "A. answer text B. answer text..."
    pattern1 = r'([A-D])\.\s*(.+?)(?=\s*[A-D]\.|$)'
    # Pattern 2: "(A) answer text (B) answer text..."
    pattern2 = r'\(([A-D])\)\s*(.+?)(?=\s*\([A-D]\)|$)'

    for pattern in [pattern1, pattern2]:
        matches = re.findall(pattern, fingertap_question, re.DOTALL)
        if matches:
            for letter, text in matches:
                choices[letter.upper()] = text.strip()
            break

    return choices


def find_matching_question(idavidrein_row, fingertap_df):
    """Find matching question in fingertap dataset."""
    idavidrein_q = normalize_text(idavidrein_row["Question"])

    best_match = None
    best_score = 0

    for idx, ft_row in fingertap_df.iterrows():
        ft_stem = normalize_text(extract_question_stem(ft_row["question"]))

        # Check if questions match (using prefix matching)
        min_len = min(len(idavidrein_q), len(ft_stem), 200)
        if min_len < 50:
            continue

        if idavidrein_q[:min_len] == ft_stem[:min_len]:
            score = min_len
            if score > best_score:
                best_score = score
                best_match = (idx, ft_row)

    return best_match


def fuzzy_match(text1: str, text2: str, min_chars: int = 10) -> bool:
    """Check if two texts match (prefix comparison with flexibility)."""
    if not text1 or not text2:
        return False

    # Compare using prefix
    min_len = min(len(text1), len(text2), 100)

    # For very short answers, require exact match
    if min_len < min_chars:
        return text1 == text2

    return text1[:min_len] == text2[:min_len]


def compare_answers(idavidrein_row, fingertap_row, fingertap_choices):
    """Compare correct and incorrect answers between sources."""
    results = {
        "correct_match": False,
        "incorrect_match": False,
        "details": {}
    }

    # Get Idavidrein answers
    id_correct = normalize_text(idavidrein_row["Correct Answer"])
    id_incorrect = [
        normalize_text(idavidrein_row.get("Incorrect Answer 1", "")),
        normalize_text(idavidrein_row.get("Incorrect Answer 2", "")),
        normalize_text(idavidrein_row.get("Incorrect Answer 3", "")),
    ]
    id_incorrect = [a for a in id_incorrect if a]  # Remove empty

    # Get fingertap correct answer
    ft_answer_letter = fingertap_row["answer"].upper()
    ft_correct = normalize_text(fingertap_choices.get(ft_answer_letter, ""))

    # Get fingertap incorrect answers
    ft_incorrect = []
    for letter, text in fingertap_choices.items():
        if letter != ft_answer_letter:
            ft_incorrect.append(normalize_text(text))

    # Compare correct answers
    if fuzzy_match(id_correct, ft_correct, min_chars=5):
        results["correct_match"] = True

    # Compare incorrect answers
    # Check if all Idavidrein incorrect answers appear in fingertap
    matched_incorrect = 0
    for id_inc in id_incorrect:
        for ft_inc in ft_incorrect:
            if fuzzy_match(id_inc, ft_inc, min_chars=5):
                matched_incorrect += 1
                break

    if len(id_incorrect) > 0 and matched_incorrect == len(id_incorrect):
        results["incorrect_match"] = True

    results["details"] = {
        "id_correct_preview": id_correct[:80],
        "ft_correct_preview": ft_correct[:80],
        "ft_answer_letter": ft_answer_letter,
        "id_incorrect_count": len(id_incorrect),
        "ft_incorrect_count": len(ft_incorrect),
        "matched_incorrect": matched_incorrect,
    }

    return results


def verify_datasets():
    """Main verification function."""
    print("=" * 70)
    print("GPQA Dataset Verification: Idavidrein vs Fingertap")
    print("=" * 70)

    # Load datasets
    df_idavidrein = load_idavidrein_gpqa()
    df_fingertap = load_fingertap_gpqa()

    # Print column info
    print("\n" + "=" * 70)
    print("COLUMN MAPPING")
    print("=" * 70)

    print("\nIdavidrein/gpqa columns:")
    print(f"  {list(df_idavidrein.columns)}")

    print("\nfingertap/GPQA-Diamond columns:")
    print(f"  {list(df_fingertap.columns)}")

    print("\n" + "-" * 70)
    print("Columns used for comparison:")
    print("-" * 70)
    print(f"  {'Field':<25} {'Idavidrein':<25} {'Fingertap':<25}")
    print(f"  {'-'*25} {'-'*25} {'-'*25}")
    print(f"  {'Question':<25} {'Question':<25} {'question':<25}")
    print(f"  {'Correct Answer':<25} {'Correct Answer':<25} {'answer (letter) + extract':<25}")
    print(f"  {'Incorrect Answer 1':<25} {'Incorrect Answer 1':<25} {'(extracted from question)':<25}")
    print(f"  {'Incorrect Answer 2':<25} {'Incorrect Answer 2':<25} {'(extracted from question)':<25}")
    print(f"  {'Incorrect Answer 3':<25} {'Incorrect Answer 3':<25} {'(extracted from question)':<25}")

    print("\nNote: Fingertap embeds answer choices in 'question' field as 'A. ... B. ... C. ... D. ...'")
    print("      The 'answer' field contains only the letter (A/B/C/D)")

    print("\n" + "=" * 70)
    print("Matching questions...")
    print("=" * 70)

    stats = {
        "total_idavidrein": len(df_idavidrein),
        "total_fingertap": len(df_fingertap),
        "questions_matched": 0,
        "questions_not_found": 0,
        "correct_answer_match": 0,
        "correct_answer_mismatch": 0,
        "incorrect_answers_match": 0,
        "incorrect_answers_mismatch": 0,
    }

    mismatches = []

    for idx, id_row in df_idavidrein.iterrows():
        match = find_matching_question(id_row, df_fingertap)

        if match is None:
            stats["questions_not_found"] += 1
            continue

        stats["questions_matched"] += 1
        ft_idx, ft_row = match

        # Extract answer choices from fingertap
        ft_choices = extract_answer_choices(ft_row["question"])

        if not ft_choices:
            print(f"  Warning: Could not extract choices for question {idx}")
            continue

        # Compare answers
        comparison = compare_answers(id_row, ft_row, ft_choices)

        if comparison["correct_match"]:
            stats["correct_answer_match"] += 1
        else:
            stats["correct_answer_mismatch"] += 1
            mismatches.append({
                "idavidrein_idx": idx,
                "fingertap_idx": ft_idx,
                "type": "correct_answer",
                "details": comparison["details"],
                "question_preview": id_row["Question"][:100],
            })

        if comparison["incorrect_match"]:
            stats["incorrect_answers_match"] += 1
        else:
            stats["incorrect_answers_mismatch"] += 1
            if comparison["correct_match"]:  # Only log if correct matched but incorrect didn't
                mismatches.append({
                    "idavidrein_idx": idx,
                    "fingertap_idx": ft_idx,
                    "type": "incorrect_answers",
                    "details": comparison["details"],
                    "question_preview": id_row["Question"][:100],
                })

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nDataset sizes:")
    print(f"  Idavidrein/gpqa:        {stats['total_idavidrein']}")
    print(f"  fingertap/GPQA-Diamond: {stats['total_fingertap']}")

    print(f"\nQuestion matching:")
    print(f"  ✓ Matched:    {stats['questions_matched']}")
    print(f"  ✗ Not found:  {stats['questions_not_found']}")

    print(f"\nCorrect answer verification (of {stats['questions_matched']} matched):")
    print(f"  ✓ Match:      {stats['correct_answer_match']} ({100*stats['correct_answer_match']/max(1,stats['questions_matched']):.1f}%)")
    print(f"  ✗ Mismatch:   {stats['correct_answer_mismatch']} ({100*stats['correct_answer_mismatch']/max(1,stats['questions_matched']):.1f}%)")

    print(f"\nIncorrect answers verification (of {stats['questions_matched']} matched):")
    print(f"  ✓ Match:      {stats['incorrect_answers_match']} ({100*stats['incorrect_answers_match']/max(1,stats['questions_matched']):.1f}%)")
    print(f"  ✗ Mismatch:   {stats['incorrect_answers_mismatch']} ({100*stats['incorrect_answers_mismatch']/max(1,stats['questions_matched']):.1f}%)")

    if mismatches:
        print(f"\n" + "=" * 70)
        print(f"MISMATCH EXAMPLES (showing first 5)")
        print("=" * 70)

        for i, mm in enumerate(mismatches[:5]):
            print(f"\n--- Mismatch {i+1} ({mm['type']}) ---")
            print(f"Question: {mm['question_preview']}...")
            print(f"Idavidrein idx: {mm['idavidrein_idx']}, Fingertap idx: {mm['fingertap_idx']}")
            d = mm['details']
            if mm['type'] == 'correct_answer':
                print(f"  Idavidrein correct: {d['id_correct_preview']}...")
                print(f"  Fingertap correct ({d['ft_answer_letter']}): {d['ft_correct_preview']}...")
            else:
                print(f"  Idavidrein incorrect count: {d['id_incorrect_count']}")
                print(f"  Fingertap incorrect count: {d['ft_incorrect_count']}")
                print(f"  Matched: {d['matched_incorrect']}/{d['id_incorrect_count']}")

    return stats, mismatches


if __name__ == "__main__":
    stats, mismatches = verify_datasets()
