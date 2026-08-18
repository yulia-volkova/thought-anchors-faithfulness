"""CoT-enabled hiring generations (the bridge experiment, generation phase).

Takes the logged direct-answer prompts (names + college conditions,
Gemma-3-12B) and swaps the answer instruction for the authors' canonical CoT
format. Generates n samples per record with vLLM.
"""
import pickle, re, sys, glob, os
sys.path.insert(0, "llm_bias")
import pandas as pd
from vllm import LLM, SamplingParams

TAIL_OLD = re.compile(r"Respond with 'Yes' or 'No' only\.\s*Respond on the next line\.\s*Answer:\s*$")
TAIL_NEW = ("Respond strictly following this format:\n\n"
            "Chain of thought: <your concise 1-2 sentence reasoning here>\nAnswer: <Yes or No>")
N = 8
MAX_LEN = 8192
llm = LLM(model="google/gemma-3-12b-it", max_model_len=MAX_LEN, gpu_memory_utilization=0.9)
tokz = llm.get_tokenizer()
sp = SamplingParams(n=N, temperature=0.7, top_p=0.95, max_tokens=200)
rows, done_conds = [], set()
if os.path.exists("hiring_cot.parquet"):
    prev = pd.read_parquet("hiring_cot.parquet")
    rows = prev.to_dict("records")
    done_conds = set(prev["cond"].unique())
    print(f"resuming: {len(rows)} rows, done conds {done_conds}", flush=True)
for cond, pat in [("names", "pkls/score_results_v3_*.pkl"),
                  ("college", "pkls_college/score_results_v3_*.pkl")]:
    if cond in done_conds:
        continue
    f = glob.glob(pat)[0]
    with open(f, "rb") as fh:
        recs = pickle.load(fh)["results"]
    prompts, meta, n_skipped = [], [], 0
    for i, r in enumerate(recs):
        p2 = TAIL_OLD.sub(TAIL_NEW, r["prompt"])
        if p2 == r["prompt"]:
            p2 = r["prompt"].rstrip().rstrip("Answer:").rstrip() + "\n\n" + TAIL_NEW
        if len(tokz(p2)["input_ids"]) > MAX_LEN - 220:
            n_skipped += 1
            continue
        prompts.append(p2)
        meta.append((i, r.get("name"), r.get("race"), r.get("gender")))
    print(f"{cond}: {len(prompts)} prompts, {n_skipped} skipped as too long", flush=True)
    out = llm.generate(prompts, sp)
    ANS = re.compile(r"Answer:\s*(Yes|No)", re.I)
    for (i, nm, race, gen_), o in zip(meta, out):
        for j, c in enumerate(o.outputs):
            m = ANS.search(c.text)
            rows.append(dict(cond=cond, idx=i, ridx=j, name=nm, race=race,
                             gender=gen_, answer=m.group(1).title() if m else None,
                             cot=c.text))
    pd.DataFrame(rows).to_parquet("hiring_cot.parquet", index=False)
    print(f"{cond}: done, total rows {len(rows)}", flush=True)
print("done")
