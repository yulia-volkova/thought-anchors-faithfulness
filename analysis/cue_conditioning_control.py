"""Cue-conditioning control (Daria/Riya feedback).

Checks whether the faithful vs unfaithful kurtosis signal depends on cue
presence, using existing data only:

1. Cue attention sanity check: does the cue sentence receive elevated attention
   in cued rollouts (with-prompt mode)?
2. Cue-excluded kurtosis: recompute with-prompt kurtosis with the cue sentence
   removed from the attention distribution; does the faithful/unfaithful gap
   survive?
3. Uncued control: compute the same faithful-PIs vs unfaithful-PIs kurtosis gap
   on UNCUED rollouts. If the gap persists without any cue, it is a problem
   property, not a faithfulness signature.
4. Receiver-head overlap: top-K receiver heads cued vs uncued per PI.

Unit of analysis is the problem instance (PI): per-head kurtosis is computed
per rollout, averaged over heads' top-K per rollout, then averaged over the
PI's rollouts. Group comparisons use Mann-Whitney U over PI means.
"""

import json
import os

import numpy as np
from scipy import stats

DATASET_CONFIGS = {
    "mmlu": {
        "dir": "final/mmlu",
        "faithful_pis": [91, 152, 188],
        "unfaithful_pis": [19, 151, 182, 191],
    },
    "gpqa": {
        "dir": "final/gpqa",
        "faithful_pis": [162, 172, 129, 160, 21],
        "unfaithful_pis": [116, 101, 107, 100, 134],
    },
}

TOP_K = 20


def pi_dir(base_dir, pi):
    return os.path.join(base_dir, f"{pi}_5_5_1")


def load_verts(base_dir, pi, condition, reasoning_only):
    suffix = "_reasoning" if reasoning_only else ""
    path = os.path.join(pi_dir(base_dir, pi), f"{condition}_head2verts{suffix}.json")
    with open(path) as f:
        return json.load(f)


def load_cue_idxs(base_dir, pi):
    path = os.path.join(pi_dir(base_dir, pi), "cued", "rollout.json")
    with open(path) as f:
        ro = json.load(f)
    return ro["prompt_cue_idxs"], ro["prompt_len"]


def head_kurtosis(head2verts, exclude_idxs=None):
    """Per-head, per-rollout kurtosis. Returns dict head -> list of kurtosis."""
    out = {}
    for head, rollouts in head2verts.items():
        ks = []
        for vs in rollouts:
            vs = np.asarray(vs, dtype=float)
            if exclude_idxs is not None:
                keep = [i for i in range(len(vs)) if i not in exclude_idxs]
                vs = vs[keep]
            if len(vs) < 4 or np.all(np.isnan(vs)):
                ks.append(np.nan)
                continue
            ks.append(stats.kurtosis(vs, fisher=True, bias=True, nan_policy="omit"))
        out[head] = ks
    return out


def pi_mean_topk_kurtosis(head2kurt, top_k=TOP_K):
    """Mean over rollouts of the mean top-K head kurtosis in that rollout."""
    n_rollouts = len(next(iter(head2kurt.values())))
    per_rollout = []
    for r in range(n_rollouts):
        vals = np.array([ks[r] for ks in head2kurt.values()], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        top = np.sort(vals)[-top_k:]
        per_rollout.append(top.mean())
    if not per_rollout:
        raise ValueError("no valid rollouts")
    return float(np.mean(per_rollout))


def top_heads(head2kurt, top_k=TOP_K):
    means = {h: np.nanmean(ks) for h, ks in head2kurt.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    return set(ranked[:top_k])


def cue_attention_check(base_dir, pi):
    """Mean vert score on cue sentence vs mean on other sentences (cued, with prompt)."""
    cue_idxs, _ = load_cue_idxs(base_dir, pi)
    h2v = load_verts(base_dir, pi, "cued", reasoning_only=False)
    cue_scores, other_scores = [], []
    for rollouts in h2v.values():
        for vs in rollouts:
            vs = np.asarray(vs, dtype=float)
            for i in cue_idxs:
                if i < len(vs) and not np.isnan(vs[i]):
                    cue_scores.append(vs[i])
            others = np.delete(vs, [i for i in cue_idxs if i < len(vs)])
            others = others[~np.isnan(others)]
            if len(others):
                other_scores.append(others.mean())
    return float(np.mean(cue_scores)), float(np.mean(other_scores))


def group_compare(f_vals, u_vals):
    stat, p = stats.mannwhitneyu(f_vals, u_vals, alternative="two-sided")
    pooled_sd = np.sqrt((np.var(f_vals, ddof=1) + np.var(u_vals, ddof=1)) / 2)
    d = (np.mean(f_vals) - np.mean(u_vals)) / pooled_sd if pooled_sd > 0 else np.nan
    return np.mean(f_vals), np.mean(u_vals), d, p


def main():
    for ds, cfg in DATASET_CONFIGS.items():
        base = cfg["dir"]
        all_pis = cfg["faithful_pis"] + cfg["unfaithful_pis"]
        labels = {pi: "F" for pi in cfg["faithful_pis"]}
        labels.update({pi: "U" for pi in cfg["unfaithful_pis"]})

        print(f"\n{'=' * 70}\nDATASET: {ds}  (F={len(cfg['faithful_pis'])}, "
              f"U={len(cfg['unfaithful_pis'])} PIs, top-{TOP_K} heads)\n{'=' * 70}")

        # 1. Cue attention sanity check
        print("\n[1] Cue sentence attention (cued, with prompt): cue vs other-sentence mean")
        for pi in all_pis:
            cue_m, other_m = cue_attention_check(base, pi)
            ratio = cue_m / other_m if other_m else np.nan
            print(f"  PI {pi:>3} ({labels[pi]}): cue={cue_m:.4f} other={other_m:.4f} ratio={ratio:.2f}")

        # Collect PI-level kurtosis under each metric variant
        variants = {
            "with-prompt (full)": dict(condition="cued", reasoning_only=False, exclude_cue=False),
            "with-prompt (cue-excluded)": dict(condition="cued", reasoning_only=False, exclude_cue=True),
            "reasoning-only cued": dict(condition="cued", reasoning_only=True, exclude_cue=False),
            "reasoning-only UNCUED": dict(condition="uncued", reasoning_only=True, exclude_cue=False),
            "with-prompt UNCUED": dict(condition="uncued", reasoning_only=False, exclude_cue=False),
        }

        results = {}
        for name, v in variants.items():
            f_vals, u_vals = [], []
            for pi in all_pis:
                exclude = None
                if v["exclude_cue"]:
                    cue_idxs, _ = load_cue_idxs(base, pi)
                    exclude = set(cue_idxs)
                h2v = load_verts(base, pi, v["condition"], v["reasoning_only"])
                h2k = head_kurtosis(h2v, exclude_idxs=exclude)
                val = pi_mean_topk_kurtosis(h2k)
                (f_vals if labels[pi] == "F" else u_vals).append(val)
            results[name] = group_compare(f_vals, u_vals)

        print(f"\n[2/3] Faithful vs unfaithful mean top-{TOP_K} kurtosis (PI-level Mann-Whitney)")
        print(f"  {'variant':<28} {'F mean':>8} {'U mean':>8} {'d':>7} {'p':>7}")
        for name, (fm, um, d, p) in results.items():
            print(f"  {name:<28} {fm:>8.3f} {um:>8.3f} {d:>7.2f} {p:>7.3f}")

        # 4. Receiver-head overlap cued vs uncued (reasoning-only)
        print("\n[4] Top receiver-head overlap: cued vs uncued (reasoning-only)")
        for pi in all_pis:
            cued = top_heads(head_kurtosis(load_verts(base, pi, "cued", True)))
            uncued = top_heads(head_kurtosis(load_verts(base, pi, "uncued", True)))
            jac = len(cued & uncued) / len(cued | uncued)
            print(f"  PI {pi:>3} ({labels[pi]}): overlap {len(cued & uncued)}/{TOP_K}  jaccard={jac:.2f}")


if __name__ == "__main__":
    main()
