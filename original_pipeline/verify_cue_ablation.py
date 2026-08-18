"""Visual verification of the cue-restatement ablation.

The "ablation" is statistical, not causal: for each receiver head we have a
vector of per-sentence vert scores (attention received by each sentence).
Kurtosis is computed over that vector. The ablation deletes the entries of
CoT sentences that mention the cue and recomputes kurtosis. If the attention
spike that produces high kurtosis WERE the cue mention, deleting it would
collapse the kurtosis. These plots make that visible.

Outputs (plots/cue_ablation/):
- profiles.png: per-problem vert-score profile for the strongest receiver head
  of the faithful rollout; cue-mention sentences in orange. If orange bars are
  the tall spikes, the signal is cue salience; if the spikes are blue, it isn't.
- ranks.png: distribution of the best cue-mention sentence's attention rank
  across all (problem, head) pairs.
- ablation_scatter.png: kurtosis with vs without cue-mention sentences, one
  point per (problem, head); points on the diagonal mean the ablation changes
  nothing.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from cue_restatement_check import (
    CUE_RE,
    DATASETS,
    kurt,
    load_pair_dir,
    vert_scores_per_head,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#b5b4ae"
OUT_DIR = "plots/cue_ablation"


def collect():
    problems = []
    for ds, base in DATASETS.items():
        for name in sorted(os.listdir(base)):
            if not name.endswith("_5_5_1"):
                continue
            pi = int(name.split("_")[0])
            fai = load_pair_dir(base, pi, "faithful")
            if fai is None:
                continue
            ro, npz = fai
            prompt_len = ro["prompt_len"]
            cue_sents = [i for i, s in enumerate(ro["sentences"])
                         if i >= prompt_len and CUE_RE.search(s)]
            if not cue_sents:
                continue
            verts = vert_scores_per_head(ro, npz)
            problems.append(dict(ds=ds, pi=pi, prompt_len=prompt_len,
                                 cue_sents=cue_sents, verts=verts))
    if not problems:
        raise RuntimeError("no problems with faithful pair + cue mentions found")
    return problems


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color=GRAY, alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_profiles(problems):
    n = len(problems)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, problems):
        # strongest receiver head for this rollout = highest kurtosis
        head = max(p["verts"], key=lambda h: kurt(p["verts"][h]))
        vs = np.asarray(p["verts"][head], dtype=float)
        idx = np.arange(len(vs))
        colors = [ORANGE if i in p["cue_sents"] else BLUE for i in idx]
        ax.bar(idx, np.nan_to_num(vs), color=colors, width=0.9)
        ax.axvspan(-0.5, p["prompt_len"] - 0.5, color=GRAY, alpha=0.25, lw=0)
        k_full = kurt(vs)
        k_wo = kurt(vs, exclude=p["cue_sents"])
        ax.set_title(
            f"{p['ds']} PI {p['pi']}  {head}   "
            f"kurt {k_full:.1f} -> {k_wo:.1f} w/o cue",
            fontsize=8.5, color="#0b0b0b")
        style_axis(ax)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRAY)]
    fig.legend(handles,
               ["sentence vert score", "cue-mention sentence", "prompt region"],
               loc="lower right", frameon=False, fontsize=9, ncol=3)
    fig.suptitle("Where does receiver-head attention actually land? "
                 "(faithful rollouts, strongest head)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(f"{OUT_DIR}/profiles.png", dpi=160)
    plt.close(fig)


def plot_ranks(problems):
    ranks, tops = [], []
    for p in problems:
        for head, vs in p["verts"].items():
            vs = np.asarray(vs, dtype=float)
            valid = ~np.isnan(vs)
            if valid.sum() < 4:
                continue
            order = np.argsort(np.where(valid, vs, -np.inf))[::-1]
            r = min(int(np.where(order == c)[0][0]) + 1
                    for c in p["cue_sents"] if c < len(vs))
            ranks.append(r)
            tops.append(int(valid.sum()))
    pct = 100 * np.array(ranks) / np.array(tops)  # rank as % of sentences
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(pct, bins=20, color=BLUE, edgecolor="white", linewidth=1)
    ax.axvline(np.median(pct), color=ORANGE, linewidth=2)
    ax.annotate(f"median {np.median(pct):.0f}%", xy=(np.median(pct), ax.get_ylim()[1] * 0.9),
                xytext=(np.median(pct) + 3, ax.get_ylim()[1] * 0.9),
                color=ORANGE, fontsize=9)
    ax.set_xlabel("attention rank of best cue-mention sentence (% of sentences, 0% = top)",
                  fontsize=9)
    ax.set_ylabel("(problem, head) pairs", fontsize=9)
    n_top3 = int(np.sum(np.array(ranks) <= 3))
    ax.set_title(f"Cue-mention sentence rarely tops the attention ranking "
                 f"({n_top3}/{len(ranks)} pairs in top 3)", fontsize=10)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ranks.png", dpi=160)
    plt.close(fig)


def plot_ablation_scatter(problems):
    k_full, k_wo = [], []
    for p in problems:
        for head, vs in p["verts"].items():
            kf = kurt(vs)
            kw = kurt(vs, exclude=p["cue_sents"])
            if not (np.isnan(kf) or np.isnan(kw)):
                k_full.append(kf)
                k_wo.append(kw)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = [min(k_full + k_wo) - 2, max(k_full + k_wo) + 2]
    ax.plot(lim, lim, color=GRAY, linewidth=1, zorder=1)
    ax.scatter(k_full, k_wo, s=22, color=BLUE, alpha=0.75, zorder=2,
               edgecolors="white", linewidths=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    med_delta = np.median(100 * (np.array(k_full) - np.array(k_wo))
                          / np.array(k_full))
    ax.set_xlabel("kurtosis (all sentences)", fontsize=9)
    ax.set_ylabel("kurtosis (cue-mention sentences removed)", fontsize=9)
    ax.set_title(f"Removing cue mentions barely moves kurtosis\n"
                 f"(median change {med_delta:.1f}%, diagonal = no change)",
                 fontsize=10)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ablation_scatter.png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    problems = collect()
    print(f"{len(problems)} problems with faithful pair + cue mentions")
    plot_profiles(problems)
    plot_ranks(problems)
    plot_ablation_scatter(problems)
    print(f"wrote 3 plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
