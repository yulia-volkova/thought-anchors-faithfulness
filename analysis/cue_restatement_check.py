"""Cue-restatement check (follow-up to the Daria/Riya cue-conditioning control).

Question: in faithful rollouts, is the attention anchor just the CoT sentence
that RESTATES the cue ("the professor said...")? If so, the faithful >
unfaithful kurtosis gap reduces to cue salience acting through the CoT.

Uses the within-problem faithful/unfaithful example rollout pairs stored in
final/{ds}/{pi}_5_5_1/faithful_vs_unfaithful/{faithful,unfaithful}/
(rollout.json + attention.npz with that PI's top receiver heads).

For each faithful rollout and each stored head:
- rank of the best cue-mention CoT sentence among all sentences by vert score
- kurtosis of vert scores with and without cue-mention sentences,
  compared to the same problem's unfaithful rollout kurtosis
"""

import json
import os
import re

import numpy as np

from run_anchors_analysis import (
    avg_matrix_by_chunk,
    get_attn_vert_scores,
)

DATASETS = {"mmlu": "final/mmlu", "gpqa": "final/gpqa"}
CUE_RE = re.compile(r"professor|stanford", re.I)


def load_pair_dir(base_dir, pi, which):
    d = os.path.join(base_dir, f"{pi}_5_5_1", "faithful_vs_unfaithful", which)
    ro_path = os.path.join(d, "rollout.json")
    npz_path = os.path.join(d, "attention.npz")
    if not (os.path.exists(ro_path) and os.path.exists(npz_path)):
        return None
    with open(ro_path) as f:
        ro = json.load(f)
    return ro, np.load(npz_path)


def vert_scores_per_head(ro, npz):
    token_ranges = ro["token_ranges"]
    out = {}
    for head in npz.files:
        avg = avg_matrix_by_chunk(npz[head], token_ranges)
        out[head] = get_attn_vert_scores(avg, rank_normalize=False)
    return out


def kurt(vs, exclude=None):
    vs = np.asarray(vs, dtype=float)
    if exclude:
        vs = np.delete(vs, [i for i in exclude if i < len(vs)])
    vs = vs[~np.isnan(vs)]
    if len(vs) < 4:
        return np.nan
    from scipy.stats import kurtosis
    return kurtosis(vs, fisher=True, bias=True)


def main():
    rows = []
    for ds, base in DATASETS.items():
        for name in sorted(os.listdir(base)):
            if not name.endswith("_5_5_1"):
                continue
            pi = int(name.split("_")[0])
            fai = load_pair_dir(base, pi, "faithful")
            unf = load_pair_dir(base, pi, "unfaithful")
            if fai is None or unf is None:
                continue
            ro_f, npz_f = fai
            ro_u, npz_u = unf
            prompt_len = ro_f["prompt_len"]
            cue_sents = [i for i, s in enumerate(ro_f["sentences"])
                         if i >= prompt_len and CUE_RE.search(s)]
            if not cue_sents:
                print(f"{ds} PI {pi}: faithful rollout has no cue-mention CoT "
                      f"sentence (mislabeled example), skipping")
                continue

            v_f = vert_scores_per_head(ro_f, npz_f)
            v_u = vert_scores_per_head(ro_u, npz_u)

            ranks, k_full, k_nocue, k_unf = [], [], [], []
            for head, vs in v_f.items():
                valid = ~np.isnan(vs)
                if valid.sum() < 4:
                    continue
                order = np.argsort(np.where(valid, vs, -np.inf))[::-1]
                best_rank = min(int(np.where(order == c)[0][0]) + 1
                                for c in cue_sents if c < len(vs))
                ranks.append(best_rank)
                k_full.append(kurt(vs))
                k_nocue.append(kurt(vs, exclude=cue_sents))
                if head in v_u:
                    k_unf.append(kurt(v_u[head]))
            n_valid = int(valid.sum())
            rows.append({
                "ds": ds, "pi": pi, "n_cue_sents": len(cue_sents),
                "n_sents": n_valid,
                "best_rank": int(np.median(ranks)),
                "top3": sum(r <= 3 for r in ranks), "n_heads": len(ranks),
                "k_faithful": float(np.nanmean(k_full)),
                "k_faithful_nocue": float(np.nanmean(k_nocue)),
                "k_unfaithful": float(np.nanmean(k_unf)) if k_unf else np.nan,
            })

    print(f"\n{'ds':<5} {'pi':>4} {'cue rank (med)':>14} {'in top3':>8} "
          f"{'k_F':>7} {'k_F w/o cue':>12} {'k_U':>7}")
    for r in rows:
        print(f"{r['ds']:<5} {r['pi']:>4} "
              f"{r['best_rank']:>7}/{r['n_sents']:<6} "
              f"{r['top3']:>4}/{r['n_heads']:<3} "
              f"{r['k_faithful']:>7.2f} {r['k_faithful_nocue']:>12.2f} "
              f"{r['k_unfaithful']:>7.2f}")

    k_f = np.array([r["k_faithful"] for r in rows])
    k_fn = np.array([r["k_faithful_nocue"] for r in rows])
    k_u = np.array([r["k_unfaithful"] for r in rows])
    print(f"\nMeans over {len(rows)} problems:")
    print(f"  faithful kurtosis           {np.nanmean(k_f):.2f}")
    print(f"  faithful w/o cue mentions   {np.nanmean(k_fn):.2f}")
    print(f"  unfaithful kurtosis         {np.nanmean(k_u):.2f}")
    print(f"  drop from removing cue mentions: "
          f"{100 * (np.nanmean(k_f) - np.nanmean(k_fn)) / np.nanmean(k_f):.1f}%")


if __name__ == "__main__":
    main()
