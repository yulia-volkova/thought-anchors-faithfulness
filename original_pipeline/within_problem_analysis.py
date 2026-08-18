"""Within-problem analysis: paired kurtosis test + probe-vs-kurtosis AUC.

Inputs: extracted_within_problem/{ds}_{pi}_{ridx}.npz from
extract_within_problem.py (verts [48, 40, S], acts [16, 2, hidden], label).

Outputs:
1. Paired within-problem test: per problem, mean top-K receiver-head kurtosis
   of verbalizing rollouts minus silent rollouts; Wilcoxon signed-rank over
   problems, per dataset and pooled.
2. Rollout-level AUCs on held-out problems (GroupKFold):
   - unsupervised kurtosis (no training - AUC of the raw score)
   - supervised probe on activations (ridge logistic, per layer/position)
   - combined (probe features + kurtosis)
3. Length control: correlation of kurtosis with n_sentences.

Writes within_problem_results.json and prints the summary table.
"""

import glob
import json
import os
import re

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

IN_DIR = "extracted_within_problem"
TOP_K = 20
PRE, POST = 0, 1  # activation positions: last prompt token, last token


def load_rollouts():
    rows = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "*.npz"))):
        m = re.match(r"(\w+)_(\d+)_(\d+)\.npz", os.path.basename(path))
        ds, pi, ridx = m.group(1), int(m.group(2)), int(m.group(3))
        z = np.load(path)
        verts = z["verts"]  # [L, H, S]
        L, H, S = verts.shape
        flat = verts.reshape(L * H, S)
        with np.errstate(all="ignore"):
            kurts = np.array([
                stats.kurtosis(v[~np.isnan(v)], fisher=True, bias=True)
                if (~np.isnan(v)).sum() >= 4 else np.nan
                for v in flat
            ])
        valid = kurts[~np.isnan(kurts)]
        if len(valid) < TOP_K:
            raise ValueError(f"{path}: only {len(valid)} valid head kurtoses")
        score = float(np.sort(valid)[-TOP_K:].mean())
        rows.append(dict(
            ds=ds, pi=pi, ridx=ridx,
            label=str(z["label"]),
            kurt_score=score,
            n_sentences=int(z["n_sentences"]),
            n_tokens=int(z["n_tokens"]),
            acts=z["acts"],
        ))
    return rows


def paired_test(rows):
    out = {}
    for ds in ["mmlu", "gpqa", "pooled"]:
        sub = [r for r in rows if ds == "pooled" or r["ds"] == ds]
        diffs = []
        for (d, pi) in sorted({(r["ds"], r["pi"]) for r in sub}):
            g = [r for r in sub if r["ds"] == d and r["pi"] == pi]
            v = [r["kurt_score"] for r in g if r["label"] == "verbalizing"]
            s = [r["kurt_score"] for r in g if r["label"] == "silent"]
            if v and s:
                diffs.append(np.mean(v) - np.mean(s))
        diffs = np.array(diffs)
        w, p = stats.wilcoxon(diffs)
        d_eff = diffs.mean() / diffs.std(ddof=1)
        out[ds] = dict(n_problems=len(diffs), mean_diff=float(diffs.mean()),
                       d=float(d_eff), wilcoxon_p=float(p),
                       frac_positive=float((diffs > 0).mean()))
    return out


def kurtosis_aucs(rows):
    y = np.array([r["label"] == "verbalizing" for r in rows])
    score = np.array([r["kurt_score"] for r in rows])
    raw_auc = roc_auc_score(y, score)
    # within-problem concordance: P(verbalizing > silent | same problem)
    wins, ties, total = 0, 0, 0
    for (d, pi) in sorted({(r["ds"], r["pi"]) for r in rows}):
        g = [r for r in rows if r["ds"] == d and r["pi"] == pi]
        for rv in [r for r in g if r["label"] == "verbalizing"]:
            for rs in [r for r in g if r["label"] == "silent"]:
                total += 1
                if rv["kurt_score"] > rs["kurt_score"]:
                    wins += 1
                elif rv["kurt_score"] == rs["kurt_score"]:
                    ties += 1
    within_auc = (wins + 0.5 * ties) / total
    r_len = stats.spearmanr(score, [r["n_sentences"] for r in rows])
    return dict(raw_auc=float(raw_auc), within_problem_auc=float(within_auc),
                spearman_kurt_vs_length=float(r_len.statistic),
                spearman_p=float(r_len.pvalue))


def probe_aucs(rows, position, add_kurtosis=False):
    """GroupKFold-by-problem logistic probe per activation layer."""
    y = np.array([r["label"] == "verbalizing" for r in rows])
    groups = np.array([f"{r['ds']}_{r['pi']}" for r in rows])
    kurt = np.array([[r["kurt_score"]] for r in rows])
    n_layers = rows[0]["acts"].shape[0]
    aucs = []
    for li in range(n_layers):
        X = np.stack([r["acts"][li, position] for r in rows])
        if add_kurtosis:
            X = np.hstack([X, kurt])
        preds = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.1, max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, preds))
    return aucs


def main():
    rows = load_rollouts()
    print(f"{len(rows)} rollouts, "
          f"{len({(r['ds'], r['pi']) for r in rows})} problems")

    paired = paired_test(rows)
    kauc = kurtosis_aucs(rows)

    results = {"paired_within_problem": paired, "kurtosis": kauc}

    print("\n[1] Paired within-problem kurtosis (verbalizing - silent)")
    for ds, r in paired.items():
        print(f"  {ds:>6}: n={r['n_problems']:>2} mean_diff={r['mean_diff']:+.2f} "
              f"d={r['d']:+.2f} p={r['wilcoxon_p']:.3f} "
              f"frac_pos={r['frac_positive']:.2f}")

    print("\n[2] Unsupervised kurtosis as detector")
    print(f"  raw AUC (all rollouts):      {kauc['raw_auc']:.3f}")
    print(f"  within-problem AUC:          {kauc['within_problem_auc']:.3f}")
    print(f"  kurtosis vs length Spearman: {kauc['spearman_kurt_vs_length']:+.2f} "
          f"(p={kauc['spearman_p']:.1e})")

    for pos, name in [(POST, "post-CoT"), (PRE, "pre-CoT")]:
        aucs = probe_aucs(rows, pos)
        results[f"probe_{name}"] = aucs
        print(f"\n[3] Supervised probe ({name}), GroupKFold by problem")
        print(f"  per-layer AUC: min={min(aucs):.3f} mean={np.mean(aucs):.3f} "
              f"max={max(aucs):.3f} (best layer idx {int(np.argmax(aucs))})")

    combo = probe_aucs(rows, POST, add_kurtosis=True)
    results["probe_post_plus_kurtosis"] = combo
    best = int(np.argmax(results["probe_post-CoT"]))
    print(f"\n[4] Combined (post-CoT probe + kurtosis feature)")
    print(f"  same layer as best probe: probe={results['probe_post-CoT'][best]:.3f} "
          f"combined={combo[best]:.3f}")

    with open("within_problem_results.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print("\nsaved within_problem_results.json")


if __name__ == "__main__":
    main()
