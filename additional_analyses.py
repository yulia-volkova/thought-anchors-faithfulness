#!/usr/bin/env python3
"""
Additional Analyses for Thought Anchors Faithfulness.

This script implements:
1. Logistic regression classifier to distinguish faithful from unfaithful CoT
2. Correlation analysis: attention -> behavior
3. Response length analysis by condition
4. Cross-tabulation: Cue Attention x Faithfulness x Accuracy
5. Per-head importance analysis

Usage:
    python additional_analyses.py
"""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from variance_analysis import (
    DATASET_CONFIGS,
    TOP_K_RECEIVER_HEADS,
    find_pi_folder,
    load_head2verts,
    collect_all_kurtosis_values,
    get_all_rollout_indices,
    get_top_heads_from_kurtosis_subset,
    compute_rollout_kurtosis_stats,
)


def load_analysis_results(path='analysis_results.json'):
    """Load the analysis_results.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def filter_faithful_unfaithful(results):
    """Filter to only faithful and unfaithful conditions."""
    faithful = [r for r in results if r.get('condition') == 'faithful']
    unfaithful = [r for r in results if r.get('condition') == 'unfaithful']
    return faithful, unfaithful


# ============================================================
# TASK 4: LOGISTIC REGRESSION CLASSIFIER
# ============================================================

def extract_features_for_classification(base_dir, pi_list, condition_type, reasoning_only=False):
    """
    Extract features for each rollout for classification.

    Args:
        base_dir: Base directory for dataset
        pi_list: List of PI IDs
        condition_type: 'faithful' or 'unfaithful'
        reasoning_only: Whether to use reasoning-only data

    Returns:
        list of dicts with features for each rollout
    """
    # Get kurtosis records
    head2kurt_records = collect_all_kurtosis_values(base_dir, pi_list, "cued", reasoning_only)

    # Get rollout-level kurtosis stats
    rollout_stats = compute_rollout_kurtosis_stats(head2kurt_records)

    # Get top-5 heads for this condition (to extract their kurtosis values)
    all_indices = set(get_all_rollout_indices(head2kurt_records))
    top_heads = get_top_heads_from_kurtosis_subset(head2kurt_records, all_indices, TOP_K_RECEIVER_HEADS)
    top_heads = sorted(list(top_heads))  # Consistent ordering

    features = []
    for stat in rollout_stats:
        pi = stat['pi']
        ridx = stat['rollout_idx']

        # Get kurtosis values for top-5 heads for this rollout
        top5_kurts = []
        for head in top_heads:
            if head in head2kurt_records:
                for (p, r, k) in head2kurt_records[head]:
                    if p == pi and r == ridx:
                        top5_kurts.append(k)
                        break

        # Pad if we don't have all 5
        while len(top5_kurts) < 5:
            top5_kurts.append(0.0)

        features.append({
            'pi': pi,
            'rollout_idx': ridx,
            'condition': condition_type,
            'mean_kurt': stat['mean_kurt'],
            'max_kurt': stat['max_kurt'],
            'median_kurt': stat['median_kurt'],
            'std_kurt': stat['std_kurt'],
            'top5_kurt_1': top5_kurts[0],
            'top5_kurt_2': top5_kurts[1],
            'top5_kurt_3': top5_kurts[2],
            'top5_kurt_4': top5_kurts[3],
            'top5_kurt_5': top5_kurts[4],
        })

    return features


def add_behavior_features(features, analysis_results):
    """
    Add behavioral features from analysis_results.json to the feature set.
    """
    # Create lookup by (pi, condition)
    lookup = {}
    for r in analysis_results:
        key = (r.get('pi'), r.get('condition'))
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(r)

    for feat in features:
        pi = feat['pi']
        # Map condition type to analysis_results condition
        cond = 'faithful' if feat['condition'] == 'faithful' else 'unfaithful'

        key = (pi, cond)
        if key in lookup and lookup[key]:
            # Take average across rollouts for this PI
            records = lookup[key]
            feat['cue_sentence_attn'] = np.nanmean([r.get('cue_sentence_attn', 0) or 0 for r in records])
            feat['cue_sentence_rank'] = np.nanmean([r.get('cue_sentence_rank', 10) or 10 for r in records])
            feat['num_sentences'] = np.nanmean([r.get('num_sentences', 0) for r in records])
        else:
            feat['cue_sentence_attn'] = 0.0
            feat['cue_sentence_rank'] = 10.0
            feat['num_sentences'] = 0.0

    return features


def run_logistic_regression_classifier():
    """
    Train and evaluate a logistic regression classifier to distinguish
    faithful from unfaithful CoT using attention patterns.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION CLASSIFIER")
    print("=" * 60)

    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, confusion_matrix,
            classification_report
        )
    except ImportError:
        print("  sklearn not available, skipping classifier")
        return None

    # Load analysis results for behavioral features
    analysis_results = load_analysis_results()

    # Collect features from both datasets
    all_features = []

    for dataset_name in ['mmlu', 'gpqa']:
        config = DATASET_CONFIGS[dataset_name]
        base_dir = config['dir']

        if not os.path.exists(base_dir):
            print(f"  Warning: {base_dir} not found, skipping")
            continue

        # Get faithful features
        faithful_features = extract_features_for_classification(
            base_dir, config['faithful_pis'], 'faithful', reasoning_only=False
        )

        # Get unfaithful features
        unfaithful_features = extract_features_for_classification(
            base_dir, config['unfaithful_pis'], 'unfaithful', reasoning_only=False
        )

        all_features.extend(faithful_features)
        all_features.extend(unfaithful_features)

    # Add behavioral features
    all_features = add_behavior_features(all_features, analysis_results)

    print(f"\nTotal samples: {len(all_features)}")
    n_faithful = sum(1 for f in all_features if f['condition'] == 'faithful')
    n_unfaithful = len(all_features) - n_faithful
    print(f"  Faithful: {n_faithful}")
    print(f"  Unfaithful: {n_unfaithful}")

    if len(all_features) < 10:
        print("  Not enough samples for classification")
        return None

    # Prepare feature matrix
    feature_names = [
        'mean_kurt', 'max_kurt', 'median_kurt', 'std_kurt',
        'top5_kurt_1', 'top5_kurt_2', 'top5_kurt_3', 'top5_kurt_4', 'top5_kurt_5',
        'cue_sentence_attn', 'cue_sentence_rank', 'num_sentences'
    ]

    X = np.array([[f.get(name, 0) for name in feature_names] for f in all_features])
    y = np.array([1 if f['condition'] == 'unfaithful' else 0 for f in all_features])

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Leave-one-out cross-validation with regularized logistic regression
    print("\nRunning Leave-One-Out Cross-Validation...")

    model = LogisticRegressionCV(
        cv=5,  # Inner CV for regularization selection
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    # Get predictions using LOO
    loo = LeaveOneOut()
    y_pred_proba = np.zeros(len(y))
    y_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in loo.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred_proba[test_idx] = model.predict_proba(X_scaled[test_idx])[:, 1]
        y_pred[test_idx] = model.predict(X_scaled[test_idx])

    # Fit final model on all data for feature importances
    model.fit(X_scaled, y)

    # Evaluate
    accuracy = accuracy_score(y, y_pred)
    try:
        auc_roc = roc_auc_score(y, y_pred_proba)
    except ValueError:
        auc_roc = None

    cm = confusion_matrix(y, y_pred)

    print(f"\n--- Results ---")
    print(f"LOO Accuracy: {accuracy:.3f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Faithful', 'Unfaithful']))

    # Compute standard errors via bootstrap for regularized model
    print(f"\nFeature Statistics (bootstrap standard errors, n_boot=1000):")

    n_bootstrap = 1000
    bootstrap_coefs = []

    for _ in range(n_bootstrap):
        # Bootstrap resample
        idx = np.random.choice(len(y), len(y), replace=True)
        X_boot = X_scaled[idx]
        y_boot = y[idx]

        # Fit model on bootstrap sample
        boot_model = LogisticRegressionCV(
            cv=3, penalty='l2', solver='lbfgs', max_iter=1000, random_state=None
        )
        try:
            boot_model.fit(X_boot, y_boot)
            bootstrap_coefs.append(boot_model.coef_[0])
        except Exception:
            continue

    bootstrap_coefs = np.array(bootstrap_coefs)

    # Compute statistics
    coefs = model.coef_[0]
    std_errs = np.std(bootstrap_coefs, axis=0)
    z_values = coefs / (std_errs + 1e-10)  # Avoid division by zero
    # Two-tailed p-values from z
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

    print(f"\n{'Feature':<20} {'Coef':>10} {'Std Err':>10} {'z':>10} {'P>|z|':>10} {'Sig':>5}")
    print("-" * 70)

    # Sort by absolute z-value
    importance_order = np.argsort(np.abs(z_values))[::-1]

    feature_stats = {}
    for idx in importance_order:
        sig = "***" if p_values[idx] < 0.001 else "**" if p_values[idx] < 0.01 else "*" if p_values[idx] < 0.05 else ""
        print(f"{feature_names[idx]:<20} {coefs[idx]:>10.4f} {std_errs[idx]:>10.4f} {z_values[idx]:>10.3f} {p_values[idx]:>10.4f} {sig:>5}")

        feature_stats[feature_names[idx]] = {
            'coef': float(coefs[idx]),
            'std_err': float(std_errs[idx]),
            'z_value': float(z_values[idx]),
            'p_value': float(p_values[idx])
        }

    # Count significant features
    n_sig = sum(1 for p in p_values if p < 0.05)
    print(f"\nSignificant features (p < 0.05): {n_sig}/{len(feature_names)}")

    model_summary = {
        'n_bootstrap': n_bootstrap,
        'n_significant_features': n_sig
    }

    results = {
        'n_samples': len(all_features),
        'n_faithful': n_faithful,
        'n_unfaithful': n_unfaithful,
        'loo_accuracy': float(accuracy),
        'auc_roc': float(auc_roc) if auc_roc else None,
        'confusion_matrix': cm.tolist(),
        'feature_statistics': feature_stats,
        'model_summary': model_summary,
        'regularization_C': float(model.C_[0]) if hasattr(model, 'C_') else None
    }

    return results


# ============================================================
# TASK 5a: CORRELATION: ATTENTION -> BEHAVIOR
# ============================================================

def load_behavioral_data_from_configs():
    """
    Load behavioral data from config files and match with attention data.

    Returns a list of dicts with:
    - pi, dataset, rollout_idx
    - is_faithful (True/False)
    - mentions_cue (from config)
    """
    data = []

    for dataset_name in ['mmlu', 'gpqa']:
        config = DATASET_CONFIGS[dataset_name]
        base_dir = config['dir']

        if not os.path.exists(base_dir):
            continue

        all_pis = config['faithful_pis'] + config['unfaithful_pis']

        for pi in all_pis:
            folder = find_pi_folder(base_dir, pi)
            if not folder:
                continue

            config_path = os.path.join(folder, 'config.json')
            if not os.path.exists(config_path):
                continue

            with open(config_path, 'r') as f:
                cfg = json.load(f)

            # Determine faithfulness for each rollout
            unfaithful_indices = set(cfg.get('cued_unfaithful_indices', []))
            mention_indices = set(cfg.get('cued_professor_mention_indices', []))
            num_rollouts = cfg.get('num_rollouts_per_condition', 5)

            for ridx in range(num_rollouts):
                data.append({
                    'pi': pi,
                    'dataset': dataset_name,
                    'rollout_idx': ridx,
                    'is_faithful': ridx not in unfaithful_indices,
                    'mentions_cue': ridx in mention_indices,
                    'condition': 'faithful' if ridx not in unfaithful_indices else 'unfaithful'
                })

    return data


def analyze_attention_behavior_correlation():
    """
    Analyze correlation between attention to cue and faithfulness/mentioning.
    Uses config files to get accurate behavioral data.
    """
    print("\n" + "=" * 60)
    print("ATTENTION -> BEHAVIOR CORRELATION")
    print("=" * 60)

    # Load behavioral data from configs
    behavioral_data = load_behavioral_data_from_configs()

    # Load attention data and match
    matched_data = []

    for dataset_name in ['mmlu', 'gpqa']:
        config = DATASET_CONFIGS[dataset_name]
        base_dir = config['dir']

        if not os.path.exists(base_dir):
            continue

        all_pis = config['faithful_pis'] + config['unfaithful_pis']

        for pi in all_pis:
            folder = find_pi_folder(base_dir, pi)
            if not folder:
                continue

            # Load head2verts to get attention data
            h2v = load_head2verts(folder, "cued", reasoning_only=False)
            if h2v is None:
                continue

            # Get number of rollouts
            sample_head = list(h2v.keys())[0]
            num_rollouts = len(h2v[sample_head])

            for ridx in range(num_rollouts):
                # Find matching behavioral record
                beh = None
                for b in behavioral_data:
                    if b['pi'] == pi and b['dataset'] == dataset_name and b['rollout_idx'] == ridx:
                        beh = b
                        break

                if beh is None:
                    continue

                # Compute mean kurtosis for this rollout
                kurts = []
                for head, vs_list in h2v.items():
                    if ridx < len(vs_list):
                        vs = vs_list[ridx]
                        if len(vs) > 3:
                            k = stats.kurtosis(vs, fisher=True, bias=True, nan_policy="omit")
                            if not np.isnan(k):
                                kurts.append(k)

                if kurts:
                    matched_data.append({
                        **beh,
                        'mean_kurtosis': np.mean(kurts),
                        'max_kurtosis': np.max(kurts),
                        'median_kurtosis': np.median(kurts)
                    })

    print(f"\nTotal matched samples: {len(matched_data)}")

    if len(matched_data) < 10:
        print("  Not enough matched data")
        return None

    analysis_results = {}

    # Split by faithfulness
    faithful_kurts = np.array([d['mean_kurtosis'] for d in matched_data if d['is_faithful']])
    unfaithful_kurts = np.array([d['mean_kurtosis'] for d in matched_data if not d['is_faithful']])

    print(f"\nFaithful samples: {len(faithful_kurts)}")
    print(f"Unfaithful samples: {len(unfaithful_kurts)}")

    # Compare kurtosis distributions
    print(f"\n--- Kurtosis by Faithfulness ---")
    print(f"Faithful:   mean={np.mean(faithful_kurts):.4f}, std={np.std(faithful_kurts):.4f}")
    print(f"Unfaithful: mean={np.mean(unfaithful_kurts):.4f}, std={np.std(unfaithful_kurts):.4f}")

    if len(faithful_kurts) > 1 and len(unfaithful_kurts) > 1:
        stat, p = stats.mannwhitneyu(faithful_kurts, unfaithful_kurts, alternative='two-sided')
        print(f"Mann-Whitney U: stat={stat:.2f}, p={p:.4f}")

        # Effect size
        pooled_std = np.sqrt((np.var(faithful_kurts) + np.var(unfaithful_kurts)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(faithful_kurts) - np.mean(unfaithful_kurts)) / pooled_std
            print(f"Cohen's d: {cohens_d:.3f}")
        else:
            cohens_d = 0

        analysis_results['kurtosis_by_faithfulness'] = {
            'faithful_mean': float(np.mean(faithful_kurts)),
            'faithful_std': float(np.std(faithful_kurts)),
            'unfaithful_mean': float(np.mean(unfaithful_kurts)),
            'unfaithful_std': float(np.std(unfaithful_kurts)),
            'mann_whitney_p': float(p),
            'cohens_d': float(cohens_d)
        }

    # Correlation with mentions_cue
    mentions_cue = np.array([1 if d['mentions_cue'] else 0 for d in matched_data])
    mean_kurts = np.array([d['mean_kurtosis'] for d in matched_data])

    if len(np.unique(mentions_cue)) > 1:
        r, p = stats.pointbiserialr(mentions_cue, mean_kurts)
        print(f"\nKurtosis vs Mentions Cue:")
        print(f"  Point-biserial r: {r:.4f}")
        print(f"  P-value: {p:.4f}")
        sig = " *" if p < 0.05 else ""
        print(f"  {sig}")
        analysis_results['kurtosis_vs_mentions'] = {'r': float(r), 'p': float(p)}

    return analysis_results


# ============================================================
# TASK 5b: RESPONSE LENGTH ANALYSIS
# ============================================================

def analyze_response_length():
    """
    Analyze whether faithful/unfaithful responses differ in length.
    """
    print("\n" + "=" * 60)
    print("RESPONSE LENGTH ANALYSIS")
    print("=" * 60)

    results = load_analysis_results()
    faithful, unfaithful = filter_faithful_unfaithful(results)

    faithful_lengths = np.array([r.get('num_sentences', 0) for r in faithful])
    unfaithful_lengths = np.array([r.get('num_sentences', 0) for r in unfaithful])

    print(f"\nFaithful responses: n={len(faithful_lengths)}")
    print(f"  Mean sentences: {np.mean(faithful_lengths):.2f}")
    print(f"  Std: {np.std(faithful_lengths):.2f}")
    print(f"  Median: {np.median(faithful_lengths):.0f}")
    print(f"  Range: [{np.min(faithful_lengths)}, {np.max(faithful_lengths)}]")

    print(f"\nUnfaithful responses: n={len(unfaithful_lengths)}")
    print(f"  Mean sentences: {np.mean(unfaithful_lengths):.2f}")
    print(f"  Std: {np.std(unfaithful_lengths):.2f}")
    print(f"  Median: {np.median(unfaithful_lengths):.0f}")
    print(f"  Range: [{np.min(unfaithful_lengths)}, {np.max(unfaithful_lengths)}]")

    analysis_results = {
        'faithful': {
            'n': len(faithful_lengths),
            'mean': float(np.mean(faithful_lengths)),
            'std': float(np.std(faithful_lengths)),
            'median': float(np.median(faithful_lengths)),
        },
        'unfaithful': {
            'n': len(unfaithful_lengths),
            'mean': float(np.mean(unfaithful_lengths)),
            'std': float(np.std(unfaithful_lengths)),
            'median': float(np.median(unfaithful_lengths)),
        }
    }

    # Statistical test
    if len(faithful_lengths) > 1 and len(unfaithful_lengths) > 1:
        stat, p = stats.mannwhitneyu(faithful_lengths, unfaithful_lengths, alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {stat:.2f}")
        print(f"  P-value: {p:.4f}")
        sig = " *" if p < 0.05 else ""
        print(f"  {sig}")

        # Effect size (rank-biserial correlation)
        n1, n2 = len(faithful_lengths), len(unfaithful_lengths)
        r = 1 - (2 * stat) / (n1 * n2)
        print(f"  Effect size (rank-biserial r): {r:.3f}")

        analysis_results['mann_whitney'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'effect_size_r': float(r)
        }

    return analysis_results


# ============================================================
# TASK 5c: CROSS-TABULATION ANALYSIS
# ============================================================

def analyze_cross_tabulation():
    """
    Cross-tabulation: Cue Attention (High/Low) x Faithfulness x Accuracy
    """
    print("\n" + "=" * 60)
    print("CROSS-TABULATION ANALYSIS")
    print("=" * 60)

    results = load_analysis_results()
    faithful, unfaithful = filter_faithful_unfaithful(results)

    all_cued = faithful + unfaithful

    # Median split on cue attention
    cue_attns = [r.get('cue_sentence_attn', None) for r in all_cued]
    cue_attns_valid = [a for a in cue_attns if a is not None]

    if not cue_attns_valid:
        print("  No valid cue attention data")
        return None

    median_attn = np.median(cue_attns_valid)
    print(f"\nMedian cue attention: {median_attn:.4f}")

    # Create cross-tabulation
    table = defaultdict(int)

    for r in all_cued:
        cue_attn = r.get('cue_sentence_attn', None)
        if cue_attn is None:
            continue

        attn_level = 'high' if cue_attn >= median_attn else 'low'
        faithfulness = r.get('condition', 'unknown')
        accuracy = 'correct' if r.get('is_correct', False) else 'incorrect'

        table[(attn_level, faithfulness, accuracy)] += 1

    print(f"\n--- Cross-tabulation ---")
    print(f"{'Attention':<10} {'Condition':<12} {'Accuracy':<12} {'Count':<8}")
    print("-" * 45)

    for key in sorted(table.keys()):
        attn, faith, acc = key
        print(f"{attn:<10} {faith:<12} {acc:<12} {table[key]:<8}")

    # Compute conditional probabilities
    print(f"\n--- Conditional Probabilities ---")

    for faith in ['faithful', 'unfaithful']:
        for attn in ['high', 'low']:
            correct = table[(attn, faith, 'correct')]
            incorrect = table[(attn, faith, 'incorrect')]
            total = correct + incorrect
            if total > 0:
                acc_rate = correct / total
                print(f"{faith:<12} + {attn:<5} attention: {acc_rate:.1%} accurate (n={total})")

    analysis_results = {
        'median_attention': float(median_attn),
        'cross_tab': {str(k): v for k, v in table.items()}
    }

    return analysis_results


# ============================================================
# TASK 5d: PER-HEAD IMPORTANCE ANALYSIS
# ============================================================

def analyze_per_head_importance():
    """
    Analyze which heads correlate most with faithfulness.
    """
    print("\n" + "=" * 60)
    print("PER-HEAD IMPORTANCE ANALYSIS")
    print("=" * 60)

    results_by_head = defaultdict(lambda: {'faithful': [], 'unfaithful': []})

    for dataset_name in ['mmlu', 'gpqa']:
        config = DATASET_CONFIGS[dataset_name]
        base_dir = config['dir']

        if not os.path.exists(base_dir):
            continue

        # Collect for faithful
        faithful_records = collect_all_kurtosis_values(
            base_dir, config['faithful_pis'], "cued", reasoning_only=False
        )
        for head, records in faithful_records.items():
            results_by_head[head]['faithful'].extend([k for _, _, k in records])

        # Collect for unfaithful
        unfaithful_records = collect_all_kurtosis_values(
            base_dir, config['unfaithful_pis'], "cued", reasoning_only=False
        )
        for head, records in unfaithful_records.items():
            results_by_head[head]['unfaithful'].extend([k for _, _, k in records])

    print(f"\nTotal heads analyzed: {len(results_by_head)}")

    # For each head, compute:
    # 1. Mean kurtosis difference (faithful - unfaithful)
    # 2. Effect size
    # 3. P-value

    head_analyses = []

    for head, values in results_by_head.items():
        f_vals = np.array(values['faithful'])
        u_vals = np.array(values['unfaithful'])

        if len(f_vals) < 3 or len(u_vals) < 3:
            continue

        mean_diff = np.mean(f_vals) - np.mean(u_vals)

        # Effect size
        pooled_std = np.sqrt((np.var(f_vals) + np.var(u_vals)) / 2)
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
        else:
            cohens_d = 0

        # Mann-Whitney test
        stat, p = stats.mannwhitneyu(f_vals, u_vals, alternative='two-sided')

        head_analyses.append({
            'head': head,
            'layer': head[0],
            'head_idx': head[1],
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            'p_value': p,
            'n_faithful': len(f_vals),
            'n_unfaithful': len(u_vals)
        })

    # Sort by absolute effect size
    head_analyses.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

    print(f"\n--- Top 10 Most Predictive Heads (by effect size) ---")
    print(f"{'Head':<12} {'Cohen d':<10} {'P-value':<10} {'Meaning'}")
    print("-" * 50)

    for h in head_analyses[:10]:
        layer, head_idx = h['head']
        d = h['cohens_d']
        p = h['p_value']
        meaning = "faithful > unfaithful" if d > 0 else "unfaithful > faithful"
        sig = "*" if p < 0.05 else ""
        print(f"L{layer}H{head_idx:<8} {d:+.3f}     {p:.4f}{sig:<5} {meaning}")

    # Significant heads
    sig_heads = [h for h in head_analyses if h['p_value'] < 0.05]
    print(f"\nTotal significant heads (p < 0.05): {len(sig_heads)}")

    # Bonferroni correction
    n_tests = len(head_analyses)
    bonf_sig = [h for h in head_analyses if h['p_value'] < 0.05 / n_tests]
    print(f"Significant after Bonferroni (p < {0.05/n_tests:.6f}): {len(bonf_sig)}")

    analysis_results = {
        'n_heads': len(head_analyses),
        'top_10_predictive': [
            {
                'head': f"L{h['layer']}H{h['head_idx']}",
                'cohens_d': float(h['cohens_d']),
                'p_value': float(h['p_value'])
            }
            for h in head_analyses[:10]
        ],
        'n_significant_uncorrected': len(sig_heads),
        'n_significant_bonferroni': len(bonf_sig)
    }

    return analysis_results


# ============================================================
# MAIN
# ============================================================

def main():
    # Change to the correct directory
    os.chdir('/Users/yuliav/PycharmProjects/thought-anchors-faithfulness')

    all_results = {}

    # Task 4: Logistic regression classifier
    classifier_results = run_logistic_regression_classifier()
    if classifier_results:
        all_results['logistic_regression'] = classifier_results

    # Task 5a: Attention -> Behavior correlation
    attn_behavior = analyze_attention_behavior_correlation()
    if attn_behavior:
        all_results['attention_behavior_correlation'] = attn_behavior

    # Task 5b: Response length analysis
    length_analysis = analyze_response_length()
    if length_analysis:
        all_results['response_length'] = length_analysis

    # Task 5c: Cross-tabulation
    cross_tab = analyze_cross_tabulation()
    if cross_tab:
        all_results['cross_tabulation'] = cross_tab

    # Task 5d: Per-head importance
    head_importance = analyze_per_head_importance()
    if head_importance:
        all_results['per_head_importance'] = head_importance

    # Save results
    output_file = 'additional_analyses_results.json'
    print(f"\n{'='*60}")
    print(f"Saving results to {output_file}")
    print(f"{'='*60}")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ADDITIONAL ANALYSES")
    print("=" * 60)

    if 'logistic_regression' in all_results:
        lr = all_results['logistic_regression']
        print(f"\n1. Logistic Regression Classifier:")
        print(f"   LOO Accuracy: {lr['loo_accuracy']:.1%}")
        if lr['auc_roc']:
            print(f"   AUC-ROC: {lr['auc_roc']:.3f}")
        print(f"   Top feature: {max(lr['feature_importances'].items(), key=lambda x: abs(x[1]))}")

    if 'attention_behavior_correlation' in all_results:
        abc = all_results['attention_behavior_correlation']
        if 'attn_vs_follows' in abc:
            print(f"\n2. Attention -> Behavior Correlation:")
            print(f"   r = {abc['attn_vs_follows']['r']:.4f}, p = {abc['attn_vs_follows']['p']:.4f}")

    if 'response_length' in all_results:
        rl = all_results['response_length']
        print(f"\n3. Response Length:")
        print(f"   Faithful: {rl['faithful']['mean']:.1f} +/- {rl['faithful']['std']:.1f} sentences")
        print(f"   Unfaithful: {rl['unfaithful']['mean']:.1f} +/- {rl['unfaithful']['std']:.1f} sentences")
        if 'mann_whitney' in rl:
            print(f"   Mann-Whitney p = {rl['mann_whitney']['p_value']:.4f}")

    if 'per_head_importance' in all_results:
        phi = all_results['per_head_importance']
        print(f"\n4. Per-Head Importance:")
        print(f"   Significant heads (p<0.05): {phi['n_significant_uncorrected']}")
        print(f"   Significant after Bonferroni: {phi['n_significant_bonferroni']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
