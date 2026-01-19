"""
Save key plots from the accuracy_cue_correlation analysis for the webapp.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_utils import load_hf_as_df

# Output directory
OUTPUT_DIR = "../webapp/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'

def load_data():
    """Load and merge all datasets."""
    print("Loading datasets...")
    
    df_cue_summary = load_hf_as_df("yulia-volkova/mmlu-chua-cue-summary")
    df_base_summary = load_hf_as_df("yulia-volkova/mmlu-chua-base-summary")
    df_no_reasoning = load_hf_as_df("yulia-volkova/mmlu-chua-no-reasoning-summary")
    
    # Merge base and cue summaries
    df_merged = pd.merge(
        df_base_summary[["pi", "accuracy", "cue_match", "gt_match"]].rename(
            columns={"accuracy": "accuracy_base", "cue_match": "cue_match_base", "gt_match": "gt_match_base"}
        ),
        df_cue_summary[["pi", "accuracy", "cue_match", "gt_match"]].rename(
            columns={"accuracy": "accuracy_cue", "cue_match": "cue_match_cue", "gt_match": "gt_match_cue"}
        ),
        on="pi",
        how="inner",
    )
    
    # Add no-reasoning
    df_merged = pd.merge(
        df_merged,
        df_no_reasoning[["pi", "accuracy"]].rename(columns={"accuracy": "accuracy_no_reasoning"}),
        on="pi",
        how="inner",
    )
    
    # Compute derived columns
    df_merged["cue_response_gap"] = df_merged["cue_match_cue"] - df_merged["cue_match_base"]
    df_merged["accuracy_diff"] = df_merged["accuracy_cue"] - df_merged["accuracy_base"]
    
    # Bin by accuracy
    df_merged["accuracy_base_bin"] = pd.cut(
        df_merged["accuracy_base"],
        bins=5,
        labels=["Very Low\n(0-0.2)", "Low\n(0.2-0.4)", "Medium\n(0.4-0.6)", "High\n(0.6-0.8)", "Very High\n(0.8-1.0)"]
    )
    
    print(f"Loaded {len(df_merged)} problems")
    return df_merged


def plot_accuracy_comparison(df):
    """Plot 1: Median accuracy across conditions."""
    accuracies = {
        "Base\n(with CoT)": df["accuracy_base"].median(),
        "Cued\n(with CoT)": df["accuracy_cue"].median(),
        "No CoT": df["accuracy_no_reasoning"].median(),
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3b7cc4", "#c45a3b", "#5a9c6a"]
    bars = ax.bar(accuracies.keys(), accuracies.values(), color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    ax.set_ylabel("Median Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Median Accuracy Across Conditions", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, (label, value) in zip(bars, accuracies.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.1%}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: accuracy_comparison.png")


def plot_cue_following_by_accuracy(df):
    """Plot 2: Mean cue following by base accuracy bin."""
    binned_cue_follow = df.groupby("accuracy_base_bin", observed=False)["cue_match_cue"].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(binned_cue_follow)))
    bars = ax.bar(range(len(binned_cue_follow)), binned_cue_follow.values, color=colors, 
                  alpha=0.85, edgecolor='white', linewidth=2)
    
    ax.set_xticks(range(len(binned_cue_follow)))
    ax.set_xticklabels(binned_cue_follow.index, fontsize=10)
    ax.set_xlabel("Base Accuracy Level", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Cue Following Rate", fontsize=12, fontweight='bold')
    ax.set_title("Cue Following by Base Accuracy Level", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cue_following_by_accuracy.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: cue_following_by_accuracy.png")


def plot_correlation_scatter(df):
    """Plot 3: Base accuracy vs cue following scatter."""
    df_valid = df[["accuracy_base", "cue_match_cue"]].dropna()
    corr = df_valid["accuracy_base"].corr(df_valid["cue_match_cue"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_valid["accuracy_base"], df_valid["cue_match_cue"], 
               alpha=0.6, s=50, c='#c45a3b', edgecolors='white', linewidth=0.5)
    
    # Trend line
    z = np.polyfit(df_valid["accuracy_base"], df_valid["cue_match_cue"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_valid["accuracy_base"].min(), df_valid["accuracy_base"].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.7, linewidth=2, label=f'r = {corr:.3f}')
    
    ax.set_xlabel("Base Accuracy (Uncued)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cue Following Rate (Cued)", fontsize=12, fontweight='bold')
    ax.set_title(f"Base Accuracy vs Cue Following (r = {corr:.3f})", fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_cue_following.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: accuracy_vs_cue_following.png")


def main():
    df = load_data()
    
    print("\nGenerating plots...")
    plot_accuracy_comparison(df)
    plot_cue_following_by_accuracy(df)
    plot_correlation_scatter(df)
    
    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()





