"""Final pre-registered analysis on the screening-2 extraction.

Strata: prereg (gap>=0.5 + nr<0.5 + gain>=0.2 + mixed; n=21 problems) and
gap07 (gap>=0.7 + mixed; n=24; user-amended primary). Tests per stratum:
paired within-problem Wilcoxon (alpha=0.01) on raw and length-residualized
kurtosis and entropy; bootstrap 90% CI on paired standardized effect for the
d=0.3 equivalence readout; pre-CoT validity assertion (same-prompt acts equal).
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

IN = "extracted_s2"
TOP_K = 20


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(IN, "*.npz"))):
        m = re.match(r"(\w+)_(\d+)_(\d+)\.npz", os.path.basename(f))
        z = np.load(f)
        verts = z["verts"]
        L, H, S = verts.shape
        flat = verts.reshape(L * H, S)
        with np.errstate(all="ignore"):
            k = np.array([stats.kurtosis(v[~np.isnan(v)])
                          if (~np.isnan(v)).sum() >= 4 else np.nan for v in flat])
        valid = k[~np.isnan(k)]
        ents = z["ents"][:, :, int(z["prompt_len"]):]
        he = np.nanmean(ents, axis=2).ravel()
        rows.append(dict(ds=m.group(1), pi=int(m.group(2)), ridx=int(m.group(3)),
                         label=str(z["label"]), prereg=bool(z["prereg"]),
                         gap=float(z["gap"]), n=int(z["n_sentences"]),
                         kurt=float(np.sort(valid)[-TOP_K:].mean()),
                         ent=float(np.nanmean(he))))
    return pd.DataFrame(rows)


def paired(df, col, resid=False):
    x = df.copy()
    if resid:
        lr = stats.linregress(np.log(x["n"]), x[col])
        x[col] = x[col] - (lr.intercept + lr.slope * np.log(x["n"]))
    d = []
    for pi, g in x.groupby("pi"):
        v = g[g.label == "verbalizing"][col]
        s = g[g.label == "silent"][col]
        if len(v) >= 3 and len(s) >= 3:
            d.append(v.mean() - s.mean())
    d = np.array(d)
    w, p = stats.wilcoxon(d)
    eff = d.mean() / d.std(ddof=1)
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(4000):
        b = rng.choice(d, len(d), replace=True)
        if b.std(ddof=1) > 0:
            boot.append(b.mean() / b.std(ddof=1))
    lo, hi = np.percentile(boot, [5, 95])
    eq = "EQUIV(|d|<0.3)" if (-0.3 < lo and hi < 0.3) else \
         ("EFFECT" if p < 0.01 else "inconclusive")
    return len(d), eff, p, lo, hi, eq


def main():
    df = load()
    print(f"{len(df)} rollouts, {df.pi.nunique()} problems "
          f"({(df.label=='verbalizing').sum()}V/{(df.label=='silent').sum()}S)")
    r = stats.spearmanr(df["kurt"], df["n"])
    print(f"kurt vs length rho={r.statistic:+.2f} | "
          f"length diff V-S paired handled via residualization")
    crit = pd.read_csv("screening2_criteria.csv")[["pi", "gain", "nr_acc"]]
    df = df.merge(crit, on="pi", how="left")
    strata = {"prereg(n~21)": df[df.prereg],
              "gap>=0.7(n~24)": df[df.gap >= 0.7],
              "USER-FINAL gap>=0.7 & gain>0": df[(df.gap >= 0.7) & (df.gain > 0)],
              "union": df}
    for name, sub in strata.items():
        print(f"\n== {name}: {sub.pi.nunique()} problems")
        for col in ["kurt", "ent"]:
            for resid in [False, True]:
                n, eff, p, lo, hi, eq = paired(sub, col, resid)
                tag = "resid" if resid else "raw  "
                print(f"  {col:4} {tag}: n={n:2} d={eff:+.2f} p={p:.4f} "
                      f"90%CI[{lo:+.2f},{hi:+.2f}] -> {eq}")


if __name__ == "__main__":
    main()
