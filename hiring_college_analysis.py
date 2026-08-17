"""College-condition attention test (direct-answer mode).

Question: does attention from the final (decision) position to the COLLEGE
token span predict college-sensitivity of the decision (range of p(yes)
across a resume's college variants)? Mirrors hiring_analysis.py but for the
inferred-demographic cue: one pkl (v3, meta job description), variants =
same resume with the college affiliation swapped.

Inputs: the logged college pkl (yes_probs) + hiring_profile/college_attn.pkl
(attention rows from hiring_attention3.py, SPAN_MODE=college).
"""

import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

LLM_BIAS = "karvonen_assets/llm_bias"
sys.path.insert(0, LLM_BIAS)  # mypkg needed to unpickle

PKL = ("karvonen_assets/paper_data_final/score_output_college_name/"
       "gpu_forward_pass/google_gemma-3-12b-it/"
       "score_results_v3_meta_job_description_google_gemma-3-12b-it_1000_0_all.pkl")
ATTN = "hiring_profile/college_attn.pkl"
COLLEGES = ["Howard University", "Morehouse College", "Georgetown University",
            "Emory University"]
SENSITIVE_THRESHOLD = 0.05


def group_key(r):
    t = r["resume"].replace(r["name"], "<N>").replace(r["email"], "<E>")
    if r.get("pronouns"):
        t = t.replace(r["pronouns"], "<P>")
    for c in COLLEGES:
        t = t.replace(c, "<C>")
    return hash(t)


def college_of(r):
    for c in COLLEGES:
        if c in r["resume"]:
            return c
    return None


def main():
    with open(PKL, "rb") as fh:
        recs = pickle.load(fh)["results"]
    with open(ATTN, "rb") as fh:
        attn = pickle.load(fh)

    gid = {i: group_key(r) for i, r in enumerate(recs)}
    col = {i: college_of(r) for i, r in enumerate(recs)}
    rows = []
    for a in attn:
        heads = np.concatenate([np.array(v) for v in a["attn_to_name"].values()])
        rows.append(dict(gid=gid[a["idx"]], college=col[a["idx"]],
                         p=a["yes_probs"],
                         att=float(np.sort(heads)[-5:].mean())))
    df = pd.DataFrame(rows)
    print(f"{len(df)} records, {df.gid.nunique()} resume groups, "
          f"college split: {df.college.value_counts().to_dict()}")

    g = df.groupby("gid").agg(dp=("p", lambda x: x.max() - x.min()),
                              att=("att", "mean"),
                              pmean=("p", "mean"), k=("p", "size")).reset_index()
    g = g[g.k >= 3]
    r = stats.spearmanr(g.att, g.dp)
    print(f"resume-level spearman(att-to-college, college dp): {r.statistic:+.3f} "
          f"(p={r.pvalue:.2e}, n={len(g)})")

    X = np.column_stack([g.pmean, np.ones(len(g))])
    ra = g.att - X @ np.linalg.lstsq(X, g.att, rcond=None)[0]
    rd = g.dp - X @ np.linalg.lstsq(X, g.dp, rcond=None)[0]
    rp = stats.spearmanr(ra, rd)
    print(f"partial (| mean p(yes)): {rp.statistic:+.3f} (p={rp.pvalue:.2e})")

    sens = g.dp > SENSITIVE_THRESHOLD
    if sens.nunique() > 1:
        auc = roc_auc_score(sens, -g.att)
        print(f"exploratory AUC (low attention -> sensitive, threshold "
              f"{SENSITIVE_THRESHOLD}): {auc:.3f} "
              f"[{int(sens.sum())} sensitive / {int((~sens).sum())} not]")

    hbcu = df[df.college.isin(["Howard University", "Morehouse College"])]
    other = df[df.college.isin(["Georgetown University", "Emory University"])]
    print(f"mean p(yes): HBCU {hbcu.p.mean():.3f} vs other {other.p.mean():.3f} | "
          f"mean att-to-college: HBCU {hbcu.att.mean():.4f} vs other "
          f"{other.att.mean():.4f}")


if __name__ == "__main__":
    main()
