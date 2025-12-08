import hashlib
import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = os.getenv(
    "LOCAL_MODEL_PATH",
    "deepseek-ai/deepseek-r1-distill-qwen-14b",
)

_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _load_model(model_name: str):
    """
    Lazily load a Hugging Face CausalLM + tokenizer once and reuse.
    """
    global _MODEL, _TOKENIZER, _DEVICE

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    print(f"[hf] Loading model '{model_name}'...")
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id

    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32,
    ).to(_DEVICE)
    _MODEL.eval()

    print(f"[hf] Model loaded on '{_DEVICE}'")
    return _MODEL, _TOKENIZER, _DEVICE


def _generate_sync(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_name: str,
) -> Dict:
    """
    Synchronous generation with transformers.
    """
    model, tokenizer, device = _load_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "text": text,
        "finish_reason": "stop",
        "usage": {},
    }


async def _generate_one(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_name: str,
    max_retries: int = 3,
    verbose: bool = False,
) -> Dict:
    """
    Async wrapper around _generate_sync with a simple retry loop.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                _generate_sync,
                prompt,
                temperature,
                top_p,
                max_tokens,
                model_name,
            )
            return result
        except Exception as e:
            last_err = e
            if verbose:
                print(
                    f"[hf] Exception during generation "
                    f"(attempt {attempt+1}/{max_retries}): {e}"
                )
            await asyncio.sleep(min(2 * (2**attempt), 60))

    return {"error": f"Generation failed after {max_retries} attempts: {last_err}"}


# ---------------------------------------------------------------------
# Multiple responses + caching
# ---------------------------------------------------------------------

async def generate_multiple_responses(
    prompt: str,
    num_responses: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    provider: str = "local",          # kept for compatibility; ignored
    model: str = DEFAULT_MODEL,
    max_retries: int = 6,
    verbose: bool = False,
    check_all_good: bool = False,
    req_exist: bool = False,
) -> Dict:
    """
    Generate multiple responses for a given prompt using a HF model.

    Returns:
        {
          "prompt": ...,
          "num_responses": ...,
          "temperature": ...,
          "top_p": ...,
          "max_tokens": ...,
          "provider": "local",
          "model": ...,
          "responses": [ { "text": ..., "finish_reason": ..., "usage": ...}, ... ]
        }
    """
    # Shorten model name for cache dir
    model_str = model.replace("/", "_")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[::2]

    nr_str = f"_nr{num_responses}"

    fp_out = (
        f"response_cache/{model_str}/"
        f"t{temperature}_p{top_p}_tok{max_tokens}_ret{max_retries}"
        f"{nr_str}_{prompt_hash}.json"
    )
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)

    # Cache check
    if os.path.exists(fp_out):
        if verbose:
            print(f"{fp_out} already exists. Loading from cache...")
        with open(fp_out, "r") as f:
            data = json.load(f)
        if not check_all_good:
            return data
        all_good = all("text" in r for r in data.get("responses", []))
        if all_good:
            return data
        if verbose:
            print("Bad data in cache. Regenerating...")
    else:
        if verbose:
            print(f"{fp_out} does not exist. Generating responses...")

    if req_exist:
        # Caller only wanted to know if cache exists
        return None

    if verbose:
        print(f"Generating {num_responses} responses with model '{model}'...")

    tasks = [
        _generate_one(
            prompt,
            temperature,
            top_p,
            max_tokens,
            model,
            max_retries=max_retries,
            verbose=verbose,
        )
        for _ in range(num_responses)
    ]

    t_start = time.time()
    responses = []

    if verbose:
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=num_responses,
            desc="Generating responses",
        ):
            responses.append(await task)
    else:
        for task in asyncio.as_completed(tasks):
            responses.append(await task)

    d = {
        "prompt": prompt,
        "num_responses": num_responses,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "provider": "local",
        "model": model,
        "responses": responses,
    }

    with open(fp_out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    if verbose:
        print(f"Saved {len(responses)} responses to {fp_out}")
        print(f"Time taken: {time.time() - t_start:.2f} seconds")

    return d


async def call_generate(
    prompt: str,
    num_responses: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 4096,
    provider: str = "local",          # kept for compatibility; ignored
    model: str = DEFAULT_MODEL,
    max_retries: int = 6,
    verbose: bool = True,
    req_exist: bool = False,
) -> Dict:
    """
    Thin wrapper to keep the same API as before.
    """
    return await generate_multiple_responses(
        prompt=prompt,
        num_responses=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        provider=provider,
        model=model,
        max_retries=max_retries,
        verbose=verbose,
        req_exist=req_exist,
    )


if __name__ == "__main__":
    example_prompt = (
        "Solve this math problem step by step. You MUST put your final answer in \\boxed{}.\n"
        "Problem: What is 2 + 2?\nSolution:\n<think>\n"
    )

    asyncio.run(
        call_generate(
            prompt=example_prompt,
            num_responses=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
            model=DEFAULT_MODEL,
            max_retries=3,
            verbose=True,
        )
    )
