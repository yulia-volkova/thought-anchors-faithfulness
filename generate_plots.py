#!/usr/bin/env python3
"""
Generate plots for the attention pattern analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_results():
    """Load all result files."""
    with open('additional_analyses_results.json', 'r') as f:
        additional = json.load(f)

    with open('extended_variance_analysis_results.json', 'r') as f:
        extended = json.load(f)

    with open('variance_analysis_results.json', 'r') as f:
        variance = json.load(f)

    return additional, extended, variance


def plot_classifier_performance(additional):
    """Plot ROC curve approximation and confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    cm = np.array(additional['logistic_regression']['confusion_matrix'])
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Faithful', 'Unfaithful'])
    ax.set_yticklabels(['Faithful', 'Unfaithful'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (n=85)\nAccuracy: {additional["logistic_regression"]["loo_accuracy"]:.1%}')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20,
                   color='white' if cm[i, j] > cm.max()/2 else 'black')

    # Feature importance bar plot
    ax = axes[1]
    feat_stats = additional['logistic_regression']['feature_statistics']
    features = list(feat_stats.keys())
    coefs = [feat_stats[f]['coef'] for f in features]
    std_errs = [feat_stats[f]['std_err'] for f in features]

    # Sort by absolute coefficient
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    features = [features[i] for i in sorted_idx]
    coefs = [coefs[i] for i in sorted_idx]
    std_errs = [std_errs[i] for i in sorted_idx]

    colors = ['#d62728' if c < 0 else '#2ca02c' for c in coefs]
    y_pos = np.arange(len(features))

    ax.barh(y_pos, coefs, xerr=std_errs, color=colors, alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Coefficient (with bootstrap SE)')
    ax.set_title('Feature Importance\n(negative = predicts unfaithful)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('plot_classifier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot_classifier.png")


def plot_ingroup_consistency(extended, variance):
    """Plot in-group consistency comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = ['mmlu', 'gpqa']
    colors = {'faithful': '#2ca02c', 'unfaithful': '#d62728'}

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Get data from variance results
        v = variance[dataset]['full']['ingroup_jaccard']
        f_mean = v['faithful']['mean']
        f_std = v['faithful']['std']
        u_mean = v['unfaithful']['mean']
        u_std = v['unfaithful']['std']
        n_f = v['faithful']['n_rollouts']
        n_u = v['unfaithful']['n_rollouts']

        # Get p-value from extended results
        p_val = extended['permutation_tests'][dataset]['full']['p_value']

        # Bar plot
        x = [0, 1]
        means = [f_mean, u_mean]
        stds = [f_std, u_std]
        labels = [f'Faithful\n(n={n_f})', f'Unfaithful\n(n={n_u})']

        bars = ax.bar(x, means, yerr=stds, color=[colors['faithful'], colors['unfaithful']],
                     alpha=0.7, capsize=5, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('In-Group Jaccard Similarity')
        ax.set_title(f'{dataset.upper()}\np = {p_val:.3f}')
        ax.set_ylim(0, 1.1)

        # Add significance line if trending
        if p_val < 0.2:
            y_max = max(means) + max(stds) + 0.1
            ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
            ax.text(0.5, y_max + 0.02, f'p={p_val:.3f}', ha='center', fontsize=10)

    plt.suptitle('In-Group Consistency by Condition (Full Context)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plot_ingroup_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot_ingroup_consistency.png")


def plot_effect_sizes(extended):
    """Plot effect sizes across datasets and modes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    effect_sizes = extended['effect_sizes']
    categories = []
    d_values = []

    for dataset in ['mmlu', 'gpqa']:
        for mode in ['full', 'reasoning_only']:
            if mode in effect_sizes[dataset]:
                d = effect_sizes[dataset][mode]['cohens_d']
                categories.append(f'{dataset.upper()}\n{mode.replace("_", " ")}')
                d_values.append(d)

    y_pos = np.arange(len(categories))
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in d_values]

    bars = ax.barh(y_pos, d_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Size: In-Group Consistency Difference\n(positive = unfaithful more consistent)")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Add effect size interpretation lines
    for thresh, label in [(-0.8, 'large'), (-0.5, 'medium'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axvline(x=thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add value labels
    for i, (bar, d) in enumerate(zip(bars, d_values)):
        ax.text(d + 0.1 if d > 0 else d - 0.1, i, f'd={d:.2f}',
               ha='left' if d > 0 else 'right', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('plot_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot_effect_sizes.png")


def plot_per_head_importance(additional):
    """Plot top predictive heads."""
    fig, ax = plt.subplots(figsize=(10, 6))

    heads = additional['per_head_importance']['top_10_predictive']

    names = [h['head'] for h in heads]
    d_values = [h['cohens_d'] for h in heads]
    p_values = [h['p_value'] for h in heads]

    y_pos = np.arange(len(names))

    # Color by significance
    colors = ['#2ca02c' if p < 0.05/1880 else '#1f77b4' if p < 0.05 else '#7f7f7f'
              for p in p_values]

    bars = ax.barh(y_pos, d_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Cohen's d (faithful - unfaithful)")
    ax.set_title("Top 10 Most Predictive Attention Heads\n(green = Bonferroni sig., blue = p<0.05, gray = n.s.)")

    # Add p-value labels
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(d_values[i] + 0.02, i, f'p={p:.4f}{sig}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('plot_head_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot_head_importance.png")


def plot_sample_sizes():
    """Create a summary of sample sizes."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Sample size data
    data = {
        'MMLU Faithful': 15,
        'MMLU Unfaithful': 20,
        'GPQA Faithful': 25,
        'GPQA Unfaithful': 25,
    }

    labels = list(data.keys())
    values = list(data.values())

    colors = ['#2ca02c', '#d62728', '#2ca02c', '#d62728']
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Rollouts')
    ax.set_title('Sample Sizes by Dataset and Condition\nTotal: 85 rollouts')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val),
               va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('plot_sample_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot_sample_sizes.png")


def main():
    os.chdir('/Users/yuliav/PycharmProjects/thought-anchors-faithfulness')

    additional, extended, variance = load_results()

    print("Generating plots...")
    plot_sample_sizes()
    plot_classifier_performance(additional)
    plot_ingroup_consistency(extended, variance)
    plot_effect_sizes(extended)
    plot_per_head_importance(additional)

    print("\nAll plots saved!")


if __name__ == "__main__":
    main()
