"""Hiring analysis: reproduces every hiring statistic in WRITEUP.md.

Estimand (corrected per review finding 1): name-only sensitivity = range of
p(yes) across the 4 name variants WITHIN each prompt version, averaged over
the 4 versions. This isolates the name intervention from prompt-version
effects.

Attention score: mean of the top-5 head values of attention from the final
pre-generation query position to the name token span (global layers only),
averaged over the 16 records (4 names x 4 versions) of a resume.

Sensitive-resume threshold for the exploratory AUC: mean within-version
range > 0.05 (chosen before analysis as "moves the decision by >5 points";
class counts printed below). All statistics are resume-level (n=111);
this is a retrospective association, not a per-response detector.

Inputs: PKL_DIR (Karvonen & Marks logged results, gemma-3-12b-it,
gm_high_bar job-description condition) and hiring_full/ (attention rows from
hiring_attention.py).
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

PKL_DIR = os.environ.get(
    "HIRING_PKL_DIR",
    "/private/tmp/claude-501/-Users-yuliav/20b073fb-85cf-44aa-8178-305d52b538d8/"
    "scratchpad/karvonen_data/paper_data_final/score_output_gm_high_bar_interventions/"
    "gpu_forward_pass/google_gemma-3-12b-it")
LLM_BIAS = os.environ.get(
    "LLM_BIAS_DIR",
    "/private/tmp/claude-501/-Users-yuliav/20b073fb-85cf-44aa-8178-305d52b538d8/"
    "scratchpad/llm_bias")
sys.path.insert(0, LLM_BIAS)  # mypkg needed to unpickle
ATTN_DIR = "hiring_full"
SENSITIVE_THRESHOLD = 0.05


def group_key(r):
    t = r["resume"].replace(r["name"], "<N>").replace(r["email"], "<E>")
    if r.get("pronouns"):
        t = t.replace(r["pronouns"], "<P>")
    return hash(t)


def load():
    rows = []
    for orig in sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl"))):
        ver = os.path.basename(orig).split("_")[2]
        with open(orig, "rb") as fh:
            d = pickle.load(fh)
        gid = {i: group_key(r) for i, r in enumerate(d["results"])}
        attn_path = os.path.join(
            ATTN_DIR, os.path.basename(orig).replace(".pkl", "_attn.pkl"))
        with open(attn_path, "rb") as fh:
            attn = pickle.load(fh)
        for a in attn:
            heads = np.concatenate([np.array(v) for v in a["attn_to_name"].values()])
            rows.append(dict(
                ver=ver, gid=gid[a["idx"]], name=a["name"], race=a["race"],
                gender=a["gender"], p=a["yes_probs"],
                att=float(np.sort(heads)[-5:].mean())))
    return pd.DataFrame(rows)


def main():
    df = load()
    print(f"{len(df)} records, {df.gid.nunique()} resumes, "
          f"{df.ver.nunique()} prompt versions")

    # within-version name-only range, then average over versions
    wv = df.groupby(["gid", "ver"]).agg(
        dp=("p", lambda x: x.max() - x.min()), att=("att", "mean"),
        pmean=("p", "mean")).reset_index()
    g = wv.groupby("gid").agg(dp=("dp", "mean"), att=("att", "mean"),
                              pmean=("pmean", "mean")).reset_index()

    r = stats.spearmanr(g.att, g.dp)
    print(f"resume-level spearman(att, name-only dp): {r.statistic:+.3f} "
          f"(p={r.pvalue:.2e}, n={len(g)})")

    X = np.column_stack([g.pmean, np.ones(len(g))])
    ra = g.att - X @ np.linalg.lstsq(X, g.att, rcond=None)[0]
    rd = g.dp - X @ np.linalg.lstsq(X, g.dp, rcond=None)[0]
    rp = stats.spearmanr(ra, rd)
    print(f"partial (| mean p(yes)): {rp.statistic:+.3f} (p={rp.pvalue:.2e})")

    sens = g.dp > SENSITIVE_THRESHOLD
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(sens, -g.att)
    print(f"exploratory resume-level AUC (low attention -> sensitive, "
          f"threshold {SENSITIVE_THRESHOLD}): {auc:.3f} "
          f"[{int(sens.sum())} sensitive / {int((~sens).sum())} not]")


if __name__ == "__main__":
    main()
