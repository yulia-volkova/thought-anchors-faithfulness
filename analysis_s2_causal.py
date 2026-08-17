"""Pre-registered analysis re-run on the causal-mask-corrected extraction.

Same estimands as analysis_s2.py, computed for three vert-score variants:
  verts       - our adaptation (proximity_ignore=3, drop_first=1, raw)
  verts_paper - the paper's analysis settings (proximity_ignore=20, drop_first=10)
  verts_rank  - rank-normalized per query row (their control_depth), 3/1
Plus: entropy validity check (normalized entropy <= 1 under causal attention),
pre-CoT activation determinism check, and the receiver-head layer-depth test
(paired mean layer of top-20 kurtosis heads, V vs S) on the analysis set and
the widened set.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

DIRS = ["extracted_s2_causal", "extracted_extra_causal"]
TOP_K = 20


def kurt_top(verts):
    L, H, S = verts.shape
    flat = verts.reshape(L * H, S)
    with np.errstate(all="ignore"):
        k = np.array([stats.kurtosis(v[~np.isnan(v)])
                      if (~np.isnan(v)).sum() >= 4 else np.nan for v in flat])
    valid = np.sort(k[~np.isnan(k)])
    top = valid[-TOP_K:]
    layers = None
    if len(valid) >= TOP_K:
        idx = np.argsort(np.nan_to_num(k, nan=-np.inf))[-TOP_K:]
        layers = float((idx // H).mean())
    return (float(top.mean()) if len(top) else np.nan), layers


def load():
    rows, ent_max = [], 0.0
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            m = re.match(r"(\w+)_(\d+)_(\d+)\.npz", os.path.basename(f))
            z = np.load(f)
            ent_max = max(ent_max, float(np.nanmax(z["ents"])))
            row = dict(src=d, ds=m.group(1), pi=int(m.group(2)),
                       ridx=int(m.group(3)), label=str(z["label"]),
                       prereg=bool(z["prereg"]), gap=float(z["gap"]),
                       n=int(z["n_sentences"]))
            for key, name in [("verts", "ours"), ("verts_paper", "paper"),
                              ("verts_rank", "rank")]:
                if key in z.files:
                    k, lay = kurt_top(z[key].astype(np.float32))
                else:  # cont_* files carry only the base verts
                    k, lay = np.nan, None
                row[f"kurt_{name}"] = k
                row[f"layer_{name}"] = lay
            ents = z["ents"][:, :, int(z["prompt_len"]):]
            row["ent"] = float(np.nanmean(np.nanmean(ents, axis=2)))
            rows.append(row)
    return pd.DataFrame(rows), ent_max


def paired(df, col, resid=False):
    x = df.dropna(subset=[col]).copy()
    if resid and len(x) > 10:
        lr = stats.linregress(np.log(x["n"]), x[col])
        x[col] = x[col] - (lr.intercept + lr.slope * np.log(x["n"]))
    d = []
    for pi, g in x.groupby("pi"):
        v = g[g.label == "verbalizing"][col]
        s = g[g.label == "silent"][col]
        if len(v) >= 3 and len(s) >= 3:
            d.append(v.mean() - s.mean())
    d = np.array(d)
    if len(d) < 6:
        return len(d), np.nan, np.nan, np.nan, np.nan, "n/a"
    w, p = stats.wilcoxon(d)
    eff = d.mean() / d.std(ddof=1)
    rng = np.random.default_rng(0)
    boot = [b.mean() / b.std(ddof=1)
            for b in (rng.choice(d, len(d), replace=True) for _ in range(4000))
            if b.std(ddof=1) > 0]
    lo, hi = np.percentile(boot, [5, 95])
    eq = "EQUIV(|d|<0.3)" if (-0.3 < lo and hi < 0.3) else \
         ("EFFECT" if p < 0.01 else "inconclusive")
    return len(d), eff, p, lo, hi, eq


def main():
    df, ent_max = load()
    print(f"{len(df)} rollouts, {df.pi.nunique()} problems "
          f"({(df.label=='verbalizing').sum()}V/{(df.label=='silent').sum()}S)")
    print(f"VALIDITY: max normalized entropy {ent_max:.3f} "
          f"({'OK, causal' if ent_max <= 1.02 else 'FAIL - not causal'})")
    for v in ["ours", "paper", "rank"]:
        c = f"kurt_{v}"
        sub = df.dropna(subset=[c])
        r = stats.spearmanr(sub[c], sub["n"])
        print(f"kurt_{v}: n={len(sub)} rho(kurt, length)={r.statistic:+.2f}")

    crit = pd.read_csv("screening2_criteria.csv")[["pi", "gain", "nr_acc"]]
    df = df.merge(crit, on="pi", how="left")
    s2 = df[df.src == "extracted_s2_causal"]
    strata = {"prereg": s2[s2.prereg], "gap>=0.7": s2[s2.gap >= 0.7],
              "gap>=0.7 & gain>0": s2[(s2.gap >= 0.7) & (s2.gain > 0)],
              "union(36)": s2}
    cols = ["kurt_ours", "kurt_paper", "kurt_rank", "ent"]
    for name, sub in strata.items():
        print(f"\n== {name}: {sub.pi.nunique()} problems")
        for col in cols:
            for resid in [False, True]:
                n, eff, p, lo, hi, eq = paired(sub, col, resid)
                tag = "resid" if resid else "raw  "
                if np.isnan(eff):
                    print(f"  {col:10} {tag}: n={n:2} -> insufficient pairs")
                else:
                    print(f"  {col:10} {tag}: n={n:2} d={eff:+.2f} p={p:.4f} "
                          f"90%CI[{lo:+.2f},{hi:+.2f}] -> {eq}")

    print("\n== receiver-head layer depth (paired V-S, top-20 kurtosis heads)")
    for v in ["ours", "paper", "rank"]:
        for scope, sub in [("analysis36", s2), ("widened", df)]:
            n, eff, p, lo, hi, eq = paired(sub, f"layer_{v}")
            if not np.isnan(eff):
                print(f"  {v:5} {scope}: n={n} d={eff:+.2f} p={p:.4f}")


if __name__ == "__main__":
    main()
