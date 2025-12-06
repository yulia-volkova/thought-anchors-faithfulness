from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER = None
MODEL = None
DEVICE: Optional[str] = None


def init_model(
    model_name: str = "deepseek-ai/deepseek-r1-distill-qwen-14b",
    device: Optional[str] = None,
) -> None:
    """
    Initialize global tokenizer & model for local generation.
    Call this once at the start.
    """
    global TOKENIZER, MODEL, DEVICE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = device

    print(f"Loading model {model_name} on {DEVICE}...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    print("Model loaded.")


def get_tokenizer():
    if TOKENIZER is None:
        raise RuntimeError("TOKENIZER not initialized. Call init_model() first.")
    return TOKENIZER


def local_generate(
    prompt: str,
    num_responses: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Local equivalent of an API sampler:
    returns {"responses": [{"text": ...}, ...]}.
    """
    if MODEL is None or TOKENIZER is None or DEVICE is None:
        raise RuntimeError("Model/tokenizer not initialized. Call init_model() first.")

    MODEL.eval()
    with torch.no_grad():
        enc = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        input_ids = enc["input_ids"]
        input_ids = input_ids.repeat(num_responses, 1)

        outputs = MODEL.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=TOKENIZER.eos_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    responses = []
    for i in range(outputs.shape[0]):
        new_tokens = outputs[i, prompt_len:]
        text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
        responses.append({"text": text})

    return {"responses": responses}
