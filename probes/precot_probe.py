"""Pre-CoT probe re-run on fixed-extraction activations (no GPU needed).

Pre-CoT activations of same-problem rollouts differ (mid-layer cosine
~0.9) despite identical prompt tokens; causal attention makes true leakage
from the continuation impossible if the kernel is correct, so the expected
cause is bf16 numeric chaos (content-independent). Diagnostic: after
problem-mean centering, pre-CoT features should NOT separate silent vs
verbalizing rollouts (AUC ~0.5 = chaos; >0.55 = leak, files unusable).

Main analysis (valid under chaos): problem-level propensity probe.
Feature = problem-mean pre-CoT activation; target = majority-silent among
followers (median split of silent share). Leave-one-problem-out logistic
AUC per layer. n = 56 problems, so exploratory either way.

Contrast: rollout-level post-CoT probe with problem-held-out CV (the
text-derivable upper bound from finding 5).
"""

import glob
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

files, pis, labels = [], [], []
for f in sorted(glob.glob("extracted_s2/*.npz") + glob.glob("extracted_extra/*.npz")):
    m = re.search(r"_(\d+)_(\d+)\.npz$", f)
    files.append(f)
    pis.append(int(m.group(1)))
pis = np.array(pis)

pre, post, labs = [], [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    pre.append(d["acts"][:, 0, :])
    post.append(d["acts"][:, 1, :])
    labs.append(str(d["label"]))
pre = np.array(pre, dtype=np.float32)   # [N, 16, 5120]
post = np.array(post, dtype=np.float32)
y = (np.array(labs) == "silent").astype(int)
act_layers = np.load(files[0])["act_layers"]
print(f"{len(files)} rollouts, {len(set(pis.tolist()))} problems, "
      f"silent {y.mean():.2f}")


def problem_held_out_auc(X, y, groups, n_folds=8):
    upi = np.unique(groups)
    folds = np.array_split(rng.permutation(upi), n_folds)
    scores = np.zeros(len(y), dtype=float)
    for fold in folds:
        te = np.isin(groups, fold)
        sc = StandardScaler().fit(X[~te])
        clf = LogisticRegression(max_iter=2000, C=0.01)
        clf.fit(sc.transform(X[~te]), y[~te])
        scores[te] = clf.decision_function(sc.transform(X[te]))
    return roc_auc_score(y, scores)


print("\n1) LEAK DIAGNOSTIC: problem-centered pre-CoT, rollout-level")
for i in [2, 5, 9, 12, 15]:
    X = pre[:, i, :].copy()
    for p in np.unique(pis):
        X[pis == p] -= X[pis == p].mean(0)
    auc = problem_held_out_auc(X, y, pis)
    print(f"  layer {act_layers[i]:2d}: centered AUC {auc:.3f}")

print("\n2) PROBLEM-LEVEL PRE-CoT PROPENSITY PROBE (LOO)")
upi = np.unique(pis)
share = np.array([y[pis == p].mean() for p in upi])
yb = (share > np.median(share)).astype(int)
print(f"  {len(upi)} problems, median silent share {np.median(share):.2f}, "
      f"class balance {yb.mean():.2f}")
for i in range(0, 16, 2):
    Xp = np.array([pre[pis == p, i, :].mean(0) for p in upi])
    scores = np.zeros(len(upi))
    for j in range(len(upi)):
        tr = np.arange(len(upi)) != j
        sc = StandardScaler().fit(Xp[tr])
        clf = LogisticRegression(max_iter=2000, C=0.01)
        clf.fit(sc.transform(Xp[tr]), yb[tr])
        scores[j] = clf.decision_function(sc.transform(Xp[j:j + 1]))[0]
    print(f"  layer {act_layers[i]:2d}: LOO AUC {roc_auc_score(yb, scores):.3f}")

print("\n3) CONTRAST: rollout-level post-CoT, problem-held-out")
for i in [9, 12, 15]:
    auc = problem_held_out_auc(post[:, i, :], y, pis)
    print(f"  layer {act_layers[i]:2d}: AUC {auc:.3f}")
