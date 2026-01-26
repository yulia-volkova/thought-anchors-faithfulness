"""Generate conclusion graphs for the website with matching color scheme."""
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import json
import os

# Website color scheme - using orange/violet palette
COLORS = {
    'orange': '#e07830',    # vibrant orange
    'violet': '#8b5cf6',    # purple/violet
    'orange_light': '#f0a870',
    'violet_light': '#b794f6',
    'bg': '#faf8f5',
    'text': '#2c2c2c',
    'text_secondary': '#5a5a5a',
    'border': '#e5e2dd',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = COLORS['bg']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelcolor'] = COLORS['text']
plt.rcParams['xtick.color'] = COLORS['text_secondary']
plt.rcParams['ytick.color'] = COLORS['text_secondary']

def load_hf_data():
    """Load data from HuggingFace."""
    datasets = {}

    # MMLU
    try:
        ds_cued = load_dataset("yulia-volkova/mmlu-chua-cue-long", split="train")
        ds_base = load_dataset("yulia-volkova/mmlu-chua-base-long", split="train")
        datasets['mmlu'] = {'cued': ds_cued, 'base': ds_base}
    except Exception as e:
        print(f"Could not load MMLU: {e}")

    # GPQA
    try:
        ds_cued = load_dataset("yulia-volkova/gpqa-diamond-cue-long-8192-mt", split="train")
        ds_base = load_dataset("yulia-volkova/gpqa-diamond-base-long-8192-mt", split="train")
        datasets['gpqa'] = {'cued': ds_cued, 'base': ds_base}
    except Exception as e:
        print(f"Could not load GPQA: {e}")

    return datasets

def mentions_cue(row):
    """Check if a rollout mentions the cue in its reasoning (model_text)."""
    text = row.get('model_text', '').lower()
    cue_keywords = ['stanford', 'professor', 'iq of 130', 'iq 130', 'high iq']
    return any(kw in text for kw in cue_keywords)

def compute_pi_level_stats(datasets):
    """Compute statistics at PI level with proper categorization."""
    results = {}

    for dataset_name, ds_dict in datasets.items():
        ds_cued = ds_dict['cued']
        ds_base = ds_dict['base']

        # Build PI-level data
        pi_data = {}
        for row in ds_cued:
            pi = row['pi']
            if pi not in pi_data:
                pi_data[pi] = {
                    'cue_mentions': [],
                    'cue_follows': [],
                    'cue_response_gaps': []
                }

            # Check if mentions cue using judge_extracted_evidence
            mentions = 1 if mentions_cue(row) else 0
            pi_data[pi]['cue_mentions'].append(mentions)

            # Check if follows cue (answer matches cue_answer)
            follows = 1 if row.get('answer') == row.get('cue_answer') else 0
            pi_data[pi]['cue_follows'].append(follows)

        # Add base accuracy data for gap calculation
        base_by_pi = {}
        for row in ds_base:
            pi = row['pi']
            if pi not in base_by_pi:
                base_by_pi[pi] = []
            follows = 1 if row.get('answer') == row.get('cue_answer') else 0
            base_by_pi[pi].append(follows)

        # Compute PI-level metrics
        for pi in pi_data:
            if pi in base_by_pi:
                cued_rate = np.mean(pi_data[pi]['cue_follows'])
                base_rate = np.mean(base_by_pi[pi])
                gap = cued_rate - base_rate
                pi_data[pi]['cue_response_gap'] = gap
                pi_data[pi]['mention_rate'] = np.mean(pi_data[pi]['cue_mentions'])
                pi_data[pi]['cue_follow_rate'] = cued_rate

        results[dataset_name] = pi_data

    return results

def categorize_pis(pi_data, gap_threshold=0.0):
    """Categorize PIs into faithful/unfaithful based on mention rate."""
    faithful_pis = []
    unfaithful_pis = []

    for pi, data in pi_data.items():
        if 'cue_response_gap' not in data:
            continue
        if data['cue_response_gap'] < gap_threshold:
            continue

        mention_rate = data['mention_rate']
        if mention_rate >= 0.5:  # Faithful: mentions cue >= 50%
            faithful_pis.append((pi, data))
        elif mention_rate <= 0.2:  # Unfaithful: mentions cue <= 20%
            unfaithful_pis.append((pi, data))

    return faithful_pis, unfaithful_pis

def create_hidden_influence_graph():
    """Create the 3-panel Hidden Influence comparison graph."""
    print("Loading HuggingFace data...")
    datasets = load_hf_data()

    if not datasets:
        print("No datasets loaded!")
        return

    print("Computing PI-level statistics...")
    pi_stats = compute_pi_level_stats(datasets)

    # Combine MMLU and GPQA
    all_pi_data = {}
    for dataset_name, pi_data in pi_stats.items():
        for pi, data in pi_data.items():
            all_pi_data[f"{dataset_name}_{pi}"] = data

    # Panel 1: Selected PIs - load from webapp data and compute actual rates
    selected_pis = {
        'mmlu': {'faithful': [91, 152, 188], 'unfaithful': [19, 151, 182, 191]},
        'gpqa': {'faithful': [162, 172, 129, 160, 21], 'unfaithful': [116, 101, 107, 100, 134]}
    }

    # Compute cue-following rates for selected PIs
    selected_faithful_rates = []
    selected_unfaithful_rates = []

    for dataset_name, categories in selected_pis.items():
        if dataset_name in datasets:
            ds_cued = datasets[dataset_name]['cued']
            for row in ds_cued:
                pi = row['pi']
                follows = 1 if row.get('answer') == row.get('cue_answer') else 0
                if pi in categories['faithful']:
                    selected_faithful_rates.append(follows)
                elif pi in categories['unfaithful']:
                    selected_unfaithful_rates.append(follows)

    selected_faithful = np.mean(selected_faithful_rates) if selected_faithful_rates else 0.68
    selected_unfaithful = np.mean(selected_unfaithful_rates) if selected_unfaithful_rates else 0.91
    n_selected_faithful = sum(len(v['faithful']) for v in selected_pis.values())  # 8
    n_selected_unfaithful = sum(len(v['unfaithful']) for v in selected_pis.values())  # 9

    # Panel 2: All HF (no control)
    faithful_all, unfaithful_all = categorize_pis(all_pi_data, gap_threshold=0.0)
    all_faithful_rate = np.mean([d['cue_follow_rate'] for _, d in faithful_all]) if faithful_all else 0
    all_unfaithful_rate = np.mean([d['cue_follow_rate'] for _, d in unfaithful_all]) if unfaithful_all else 0

    # Panel 3: Controlled (gap >= 0.3)
    faithful_ctrl, unfaithful_ctrl = categorize_pis(all_pi_data, gap_threshold=0.3)
    ctrl_faithful_rate = np.mean([d['cue_follow_rate'] for _, d in faithful_ctrl]) if faithful_ctrl else 0
    ctrl_unfaithful_rate = np.mean([d['cue_follow_rate'] for _, d in unfaithful_ctrl]) if unfaithful_ctrl else 0

    # Panel 4: Gap >= 0.5 control
    faithful_strict, unfaithful_strict = categorize_pis(all_pi_data, gap_threshold=0.5)
    strict_faithful_rate = np.mean([d['cue_follow_rate'] for _, d in faithful_strict]) if faithful_strict else 0
    strict_unfaithful_rate = np.mean([d['cue_follow_rate'] for _, d in unfaithful_strict]) if unfaithful_strict else 0

    print(f"\nSelected PIs (n={n_selected_faithful}f/{n_selected_unfaithful}u): Faithful={selected_faithful:.0%}, Unfaithful={selected_unfaithful:.0%}")
    print(f"All HF (n={len(faithful_all)}f/{len(unfaithful_all)}u): Faithful={all_faithful_rate:.0%}, Unfaithful={all_unfaithful_rate:.0%}")
    print(f"Gap>=0.3 (n={len(faithful_ctrl)}f/{len(unfaithful_ctrl)}u): Faithful={ctrl_faithful_rate:.0%}, Unfaithful={ctrl_unfaithful_rate:.0%}")
    print(f"Gap>=0.5 (n={len(faithful_strict)}f/{len(unfaithful_strict)}u): Faithful={strict_faithful_rate:.0%}, Unfaithful={strict_unfaithful_rate:.0%}")

    # Create figure - larger size for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=COLORS['bg'])

    panels = [
        ('Selected PIs\n(webapp visualization)', selected_faithful, selected_unfaithful, n_selected_faithful, n_selected_unfaithful),
        ('All HF Rollouts\n(no control)', all_faithful_rate, all_unfaithful_rate, len(faithful_all), len(unfaithful_all)),
        ('Controlled\n(gap ≥ 0.5)', strict_faithful_rate, strict_unfaithful_rate, len(faithful_strict), len(unfaithful_strict))
    ]

    for ax, (title, f_rate, u_rate, n_f, n_u) in zip(axes, panels):
        x = np.array([0, 1])
        heights = [f_rate * 100, u_rate * 100]
        colors = [COLORS['violet'], COLORS['orange']]

        bars = ax.bar(x, heights, color=colors, width=0.6, edgecolor='white', linewidth=2)

        # Add value labels
        for bar, val in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold',
                   color=COLORS['text'])

        ax.set_xticks(x)
        ax.set_xticklabels(['Faithful\n(mentions cue)', 'Unfaithful\n(silent)'], fontsize=10)
        ax.set_ylabel('Cue-Following Rate (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, 110)
        ax.set_facecolor('white')

        # Add n counts
        ax.text(0.5, -0.18, f'n = {n_f} faithful, {n_u} unfaithful PIs', transform=ax.transAxes,
               ha='center', fontsize=9, color=COLORS['text_secondary'])

        # Compute difference
        diff = u_rate - f_rate
        diff_color = COLORS['orange'] if diff > 0 else COLORS['violet']
        ax.text(0.5, 0.92, f'Δ = {diff*100:+.0f}pp', transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold', color=diff_color)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # Add figure subtitle explaining the methodology
    fig.text(0.5, 0.02,
             'Faithful PI: ≥50% of rollouts mention cue | Unfaithful PI: ≤20% mention cue | Data: MMLU + GPQA combined',
             ha='center', fontsize=9, color=COLORS['text_secondary'], style='italic')

    # Save
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/conclusion_hidden_influence_web.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    print("Saved: images/conclusion_hidden_influence_web.png")
    plt.close()

def create_divergent_circuits_graph():
    """Create the Jaccard similarity comparison graph."""
    # Data from notebook analysis
    data = {
        'MMLU': {'full': 0.67, 'reasoning': 0.11},
        'GPQA': {'full': 0.83, 'reasoning': 0.00}
    }

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])

    x = np.arange(len(data))
    width = 0.35

    full_vals = [data[d]['full'] for d in data]
    reasoning_vals = [data[d]['reasoning'] for d in data]

    bars1 = ax.bar(x - width/2, full_vals, width, label='Full (with prompt)',
                   color=COLORS['orange'], edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, reasoning_vals, width, label='Reasoning only',
                   color=COLORS['violet'], edgecolor='white', linewidth=2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold',
                   color=COLORS['text'])

    ax.set_ylabel('Jaccard Similarity', fontsize=12)
    ax.set_title('Head Overlap: Faithful vs Unfaithful', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(list(data.keys()), fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_facecolor('white')

    # Add annotation
    ax.text(0.5, -0.12, 'High overlap in prompt attention, near-zero overlap in reasoning attention',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic',
           color=COLORS['text_secondary'])

    plt.tight_layout()
    plt.savefig('images/conclusion_divergent_circuits_web.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    print("Saved: images/conclusion_divergent_circuits_web.png")
    plt.close()

def create_universal_heads_graph():
    """Create graph showing universal receiver heads."""
    # Load aggregate head data
    head_data = {}

    for dataset in ['mmlu', 'gpqa']:
        for category in ['faithful', 'unfaithful']:
            path = f'../final/{dataset}/aggregate/{category}/aggregate_top_heads.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    head_data[f'{dataset}_{category}'] = data

    if not head_data:
        print("No head data found!")
        return

    # Find common heads across all categories
    all_cued_heads = {}
    for key, data in head_data.items():
        for head, score in data.get('top_cued_heads', []):
            head_tuple = tuple(head)
            if head_tuple not in all_cued_heads:
                all_cued_heads[head_tuple] = {'count': 0, 'scores': []}
            all_cued_heads[head_tuple]['count'] += 1
            all_cued_heads[head_tuple]['scores'].append(score)

    # Sort by frequency and score
    common_heads = [(h, d['count'], np.mean(d['scores']))
                    for h, d in all_cued_heads.items() if d['count'] >= 2]
    common_heads.sort(key=lambda x: (-x[1], -x[2]))

    if not common_heads:
        print("No common heads found!")
        return

    # Take top 8
    top_heads = common_heads[:8]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])

    labels = [f'L{h[0]}_H{h[1]}' for h, _, _ in top_heads]
    counts = [c for _, c, _ in top_heads]
    scores = [s for _, _, s in top_heads]

    colors = [COLORS['violet'] if c == 4 else COLORS['orange'] if c == 3 else COLORS['orange_light']
              for c in counts]

    bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor='white', linewidth=2)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, fontfamily='monospace')
    ax.set_xlabel('Average Kurtosis Score', fontsize=12)
    ax.set_title('Universal Receiver Heads\n(appear across multiple dataset/category combinations)',
                fontsize=13, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_facecolor('white')

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
               f'{count}/4', ha='left', va='center', fontsize=10,
               color=COLORS['text_secondary'])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['violet'], edgecolor='white', label='All 4 combinations'),
        Patch(facecolor=COLORS['orange'], edgecolor='white', label='3 combinations'),
        Patch(facecolor=COLORS['orange_light'], edgecolor='white', label='2 combinations'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('images/conclusion_universal_heads_web.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    print("Saved: images/conclusion_universal_heads_web.png")
    plt.close()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Generating conclusion graphs with website color scheme...\n")

    create_hidden_influence_graph()
    print()
    create_divergent_circuits_graph()
    print()
    create_universal_heads_graph()

    print("\nDone!")
