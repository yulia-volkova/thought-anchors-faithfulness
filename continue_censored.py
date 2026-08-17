"""Continue censored (cap-truncated) rollouts with a raised ceiling (16k)."""
import pandas as pd
from vllm import LLM, SamplingParams

m = pd.read_parquet("continuation_manifest.parquet")
llm = LLM(model="deepseek-ai/deepseek-r1-distill-qwen-14b",
          max_model_len=16384, gpu_memory_utilization=0.92)
tok = llm.get_tokenizer()
prompts, meta, params = [], [], []
for _, r in m.iterrows():
    plen = len(tok.encode(r.prefix))
    mt = 16384 - plen - 32
    if mt < 128:
        print(f"SKIP pi {r.pi} r{r.ridx}: prefix {plen}")
        continue
    prompts.append(r.prefix)
    meta.append(r)
    params.append(SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=mt))
res = llm.generate(prompts, params)
rows = []
for r, o in zip(meta, res):
    rows.append(dict(pi=int(r.pi), cond=r.cond, ridx=int(r.ridx),
                     gt_answer=r.gt_answer, cue_answer=r.cue_answer,
                     continuation=o.outputs[0].text))
pd.DataFrame(rows).to_parquet("continuations.parquet", index=False)
print(f"done: {len(rows)} continuations")
