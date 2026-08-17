"""vLLM screening generation for the powered strict rerun.

Per problem in data/screening_set.csv:
- 32 cued rollouts, 20 uncued, 8 no-reasoning
- temperature 0.7, top_p 0.95 (stage-1 settings), max_tokens 8192 (nr: 32)
Answers parsed with the canonical format regex. Output: screening1_regen.parquet
(one row per generation) written incrementally every SAVE_EVERY problems.
"""

import os
import re

import pandas as pd
from vllm import LLM, SamplingParams

MODEL = "deepseek-ai/deepseek-r1-distill-qwen-14b"
ANS_RE = re.compile(r"best answer is:?\s*\(?([A-J])\)?", re.I)
CONDS = [("cued", 32, 8192), ("uncued", 20, 8192), ("nr", 8, 32)]
SAVE_EVERY = 10
OUT = "screening1_regen.parquet"


def main():
    df = pd.read_csv("data/screening_set.csv")
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_parquet(OUT, columns=["pi"]).pi.unique())
        print(f"resuming: {len(done)} problems already done")
    llm = LLM(model=MODEL, max_model_len=12288, gpu_memory_utilization=0.92)

    buf = []
    todo = df[~df.pi.isin(done)]
    for bi in range(0, len(todo), SAVE_EVERY):
        batch = todo.iloc[bi:bi + SAVE_EVERY]
        prompts, meta = [], []
        for _, r in batch.iterrows():
            for cond, n, mt in CONDS:
                prompts.append(r[f"prompt_{cond}"])
                meta.append((r.ds, int(r.pi), cond, n, mt, r.gt_answer, r.cue_answer))
        tok = llm.get_tokenizer()
        kept_prompts, kept_meta, outs = [], [], []
        for (p, m) in zip(prompts, meta):
            plen = len(tok.encode(p))
            if plen > 11520:
                print(f"SKIP pi {m[1]} {m[2]}: prompt {plen} tokens", flush=True)
                continue
            kept_prompts.append(p)
            kept_meta.append(m)
            outs.append(SamplingParams(n=m[3], temperature=0.7, top_p=0.95,
                                       max_tokens=max(256, min(m[4], 12288 - plen - 32)),
                                       seed=None))
        prompts, meta = kept_prompts, kept_meta
        results = llm.generate(prompts, outs)
        for m, res in zip(meta, results):
            for j, o in enumerate(res.outputs):
                ans = ANS_RE.findall(o.text)
                buf.append(dict(ds=m[0], pi=m[1], cond=m[2], ridx=j,
                                gt_answer=m[5], cue_answer=m[6],
                                answer=ans[-1].upper() if ans else None,
                                text=o.text))
        new = pd.DataFrame(buf)
        if os.path.exists(OUT):
            new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
        new.to_parquet(OUT, index=False)
        buf = []
        print(f"saved through problem batch {bi + len(batch)}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
