"""Expanded within-problem analysis on v2 extraction (kurtosis + entropy).

Tests, on all available verbalizing/silent rollouts of 113 mixed problems:

1. Replication of the kurtosis result at larger n (paired test, raw and
   length-residualized).
2. The diffuseness hypothesis, properly: normalized attention entropy
   (0 = one-hot, 1 = uniform; length-invariant by construction).
   Hypothesis: silent (unfaithful) rollouts are MORE diffuse (higher entropy).
   Scores per rollout:
   - ent_all: mean normalized entropy over reasoning sentences, all heads
   - ent_focused: same, over the 20 most-focused heads (lowest entropy) -
     the heads where anchoring actually happens
3. Length sanity check for entropy (Spearman vs n_sentences).
4. AUCs: within-problem concordance for each score.

Writes entropy_results.json and rollout_scores_v2.csv.
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

IN_DIR = "extracted_v2"
TOP_K = 20


def rollout_scores():
    cache = "rollout_scores_v2.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "*.npz"))):
        m = re.match(r"(\w+)_(\d+)_(\d+)\.npz", os.path.basename(path))
        z = np.load(path)
        verts, ents = z["verts"], z["ents"]
        prompt_len = int(z["prompt_len"])
        L, H, S = verts.shape
        flat_v = verts.reshape(L * H, S)
        with np.errstate(all="ignore"):
            kurts = np.array([
                stats.kurtosis(v[~np.isnan(v)])
                if (~np.isnan(v)).sum() >= 4 else np.nan for v in flat_v])
        valid = kurts[~np.isnan(kurts)]
        kurt_score = float(np.sort(valid)[-TOP_K:].mean())

        # entropy over reasoning sentences only
        ent_reason = ents[:, :, prompt_len:]          # [L, H, S_r]
        head_ent = np.nanmean(ent_reason, axis=2).ravel()  # [L*H]
        ent_all = float(np.nanmean(head_ent))
        focused_idx = np.argsort(head_ent)[:TOP_K]
        ent_focused = float(head_ent[focused_idx].mean())

        rows.append(dict(
            ds=m.group(1), pi=int(m.group(2)), ridx=int(m.group(3)),
            label=str(z["label"]), n=int(z["n_sentences"]),
            n_tokens=int(z["n_tokens"]),
            kurt=kurt_score, ent_all=ent_all, ent_focused=ent_focused))
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def paired(df, col):
    out = {}
    for ds in ["mmlu", "gpqa", "pooled"]:
        sub = df if ds == "pooled" else df[df.ds == ds]
        diffs = []
        for (d, pi), g in sub.groupby(["ds", "pi"]):
            v = g[g.label == "verbalizing"][col]
            s = g[g.label == "silent"][col]
            if len(v) and len(s):
                diffs.append(v.mean() - s.mean())
        diffs = np.array(diffs)
        w, p = stats.wilcoxon(diffs)
        out[ds] = dict(n=len(diffs), mean_diff=float(diffs.mean()),
                       d=float(diffs.mean() / diffs.std(ddof=1)),
                       p=float(p), frac_pos=float((diffs > 0).mean()))
    return out


def within_auc(df, col, higher_is_verbalizing=True):
    wins, ties, total = 0, 0, 0
    for (d, pi), g in df.groupby(["ds", "pi"]):
        for _, rv in g[g.label == "verbalizing"].iterrows():
            for _, rs in g[g.label == "silent"].iterrows():
                total += 1
                a, b = rv[col], rs[col]
                if not higher_is_verbalizing:
                    a, b = b, a
                wins += a > b
                ties += a == b
    return (wins + 0.5 * ties) / total, total


def main():
    df = rollout_scores()
    # restrict to problems satisfying ALL original selection criteria:
    # cue != gt (enforced at selection), cue-response gap >= 0.5,
    # no-reasoning accuracy < 0.5 (problem_criteria.csv)
    crit = pd.read_csv("problem_criteria.csv")
    strict = crit[crit.strict][["ds", "pi"]]
    df = df.merge(strict, on=["ds", "pi"], how="inner")
    n_prob = df.groupby(["ds", "pi"]).ngroups
    print(f"STRICT-CRITERIA subset: {len(df)} rollouts, {n_prob} problems "
          f"({(df.label == 'verbalizing').sum()} V, {(df.label == 'silent').sum()} S)")

    results = {}
    print("\n[1] Kurtosis replication at larger n (paired within-problem)")
    results["kurt_paired"] = paired(df, "kurt")
    for ds, r in results["kurt_paired"].items():
        print(f"  {ds:>6}: n={r['n']:>3} d={r['d']:+.2f} p={r['p']:.4f} "
              f"frac_pos={r['frac_pos']:.2f}")
    lr = stats.linregress(np.log(df["n"]), np.log(df["kurt"] - df["kurt"].min() + 1))
    df["kurt_resid"] = (np.log(df["kurt"] - df["kurt"].min() + 1)
                        - (lr.intercept + lr.slope * np.log(df["n"])))
    results["kurt_resid_paired"] = paired(df, "kurt_resid")
    r = results["kurt_resid_paired"]["pooled"]
    print(f"  length-residualized pooled: d={r['d']:+.2f} p={r['p']:.3f} "
          f"frac_pos={r['frac_pos']:.2f}")

    print("\n[2] Diffuseness hypothesis: normalized attention entropy")
    for col in ["ent_all", "ent_focused"]:
        results[f"{col}_paired"] = paired(df, col)
        rho = stats.spearmanr(df[col], df["n"])
        print(f"  {col}: Spearman vs length = {rho.statistic:+.2f}")
        for ds, r in results[f"{col}_paired"].items():
            print(f"    {ds:>6}: n={r['n']:>3} mean_diff(V-S)={r['mean_diff']:+.4f} "
                  f"d={r['d']:+.2f} p={r['p']:.4f} frac_pos={r['frac_pos']:.2f}")

    print("\n[3] Within-problem AUCs")
    for col, hi in [("kurt", True), ("ent_all", False), ("ent_focused", False)]:
        auc, total = within_auc(df, col, higher_is_verbalizing=hi)
        results[f"auc_{col}"] = dict(auc=float(auc), pairs=int(total))
        direction = "V higher" if hi else "S higher (diffuse-unfaithful hypothesis)"
        print(f"  {col:<12} AUC={auc:.3f} over {total} pairs  [{direction}]")

    with open("entropy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved entropy_results.json")


if __name__ == "__main__":
    main()
