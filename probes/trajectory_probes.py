"""Trajectory-probe pilot (runs on the box next to the data).

Question: is "this rollout will verbalize the hint" readable from the
model's state during the reasoning, before the mention happens?

Design: binary probe (silent vs verbalizing followers) on per-sentence
states (act_traj). For verbalizing rollouts only sentences BEFORE the first
hint mention are used, so the probe cannot read the mention off the context.
Logistic regression per layer x relative-position bin, problem-held-out CV,
AUC. Pilot caveats: 56 problems; sentence alignment between the splitter
and saved trajectories is approximate for rollouts where empty sentences
were dropped (skipped when lengths mismatch).
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from anchors_utils import split_prompt_into_chunks, split_solution_into_chunks

DIRS = ["extracted_s2_final", "extracted_extra_final"]
MANIFESTS = ["extraction_manifest.parquet", "extraction_manifest_extra.parquet"]
MENTION = re.compile(r"professor|stanford", re.I)
LAYERS = [3, 6, 9, 12]           # indices into the 16 saved act layers
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
rng = np.random.default_rng(0)

texts = {}
for mf in MANIFESTS:
    if os.path.exists(mf):
        m = pd.read_parquet(mf)
        for _, r in m.iterrows():
            texts[(r.ds, int(r.pi), int(r.ridx))] = (r.prompt_cued, str(r.text))

rows = []
skipped = 0
for d in DIRS:
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        mm = re.match(r"(\w+)_(\d+)_(\d+)\.npz", os.path.basename(f))
        key = (mm.group(1), int(mm.group(2)), int(mm.group(3)))
        if key not in texts:
            skipped += 1
            continue
        z = np.load(f)
        if "act_traj" not in z.files:
            skipped += 1
            continue
        prompt, gen = texts[key]
        gsents = split_solution_into_chunks(gen)
        pl = int(z["prompt_len"])
        S = int(z["n_sentences"])
        if pl + len(gsents) != S:
            skipped += 1
            continue
        first_mention = next((j for j, s in enumerate(gsents) if MENTION.search(s)),
                             None)
        label = str(z["label"])
        n_gen = len(gsents)
        limit = first_mention if (label == "verbalizing" and
                                  first_mention is not None) else n_gen
        traj = z["act_traj"].astype(np.float32)  # [16, S, H]
        for j in range(limit):
            rows.append(dict(pi=key[1], label=label, relpos=j / max(1, n_gen - 1),
                             feats={li: traj[li, pl + j, :] for li in LAYERS}))
print(f"{len(rows)} sentence-states from usable rollouts, {skipped} rollouts skipped",
      flush=True)

df = pd.DataFrame([{k: v for k, v in r.items() if k != "feats"} for r in rows])
y_all = (df.label == "silent").astype(int).values
pis = df.pi.values

for li in LAYERS:
    X_all = np.stack([r["feats"][li] for r in rows])
    print(f"\nlayer index {li}:", flush=True)
    for lo, hi in BINS:
        sel = (df.relpos >= lo) & (df.relpos < hi)
        X, y, g = X_all[sel.values], y_all[sel.values], pis[sel.values]
        if len(np.unique(y)) < 2 or len(y) < 200:
            print(f"  bin {lo:.1f}-{hi:.1f}: insufficient data")
            continue
        upi = rng.permutation(np.unique(g))
        folds = np.array_split(upi, 8)
        scores = np.zeros(len(y))
        for fold in folds:
            te = np.isin(g, fold)
            if te.sum() == 0 or (~te).sum() == 0:
                continue
            sc = StandardScaler().fit(X[~te])
            clf = LogisticRegression(max_iter=1000, C=0.003)
            clf.fit(sc.transform(X[~te]), y[~te])
            scores[te] = clf.decision_function(sc.transform(X[te]))
        auc = roc_auc_score(y, scores)
        print(f"  bin {lo:.1f}-{hi:.1f}: n={len(y):5d} "
              f"({y.mean():.2f} silent) AUC={auc:.3f}", flush=True)
print("done")
