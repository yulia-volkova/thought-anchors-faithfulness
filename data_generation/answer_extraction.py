# answer_extraction.py

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    # Try exact pattern first
    pattern = r"Therefore, the best answer is: \(([^)]+)\)\."
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try more flexible patterns
    patterns = [
        r"Therefore, the best answer is:?\s*\(([^)]+)\)",
        r"the best answer is:?\s*\(([^)]+)\)",
        r"Therefore,?\s*(?:the\s*)?(?:best\s*)?answer\s*is:?\s*\(([^)]+)\)",
        r"answer\s*is:?\s*\(([^)]+)\)",
    ]

    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None
