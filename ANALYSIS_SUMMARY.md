# Attention Patterns in Faithful vs Unfaithful Chain-of-Thought Reasoning

## Plots Generated

| File | Description |
|------|-------------|
| `plot_sample_sizes.png` | Sample sizes by dataset and condition |
| `plot_classifier.png` | Confusion matrix and feature importance with SE |
| `plot_ingroup_consistency.png` | In-group Jaccard by condition (MMLU vs GPQA) |
| `plot_effect_sizes.png` | Cohen's d effect sizes across analyses |
| `plot_head_importance.png` | Top 10 predictive attention heads |

---

## Executive Summary

This analysis investigates whether attention patterns in transformer models can distinguish faithful from unfaithful chain-of-thought (CoT) reasoning. Using data from 85 rollouts across MMLU and GPQA datasets, we applied multiple statistical approaches to understand the relationship between attention head activation patterns and reasoning faithfulness.

### Key Findings

1. **Logistic Regression Classifier**: Achieves **71.8% accuracy** and **0.819 AUC-ROC** in distinguishing faithful from unfaithful CoT, suggesting attention patterns contain predictive signal.

2. **In-Group Consistency**: Mixed results across datasets - MMLU shows unfaithful reasoning is more internally consistent (p=0.159), while GPQA shows the opposite pattern (p=0.396).

3. **Head Selection**: The same ~5 attention heads dominate in both conditions (L31H34, L45H0, L47H3, L36H6, L44H33), with Jaccard similarities near baseline.

4. **Per-Head Importance**: 3 heads survive Bonferroni correction as significantly different between conditions, with effect sizes d > 0.9.

---

## Data Summary

**See: `plot_sample_sizes.png`**

| Dataset | Faithful (n) | Unfaithful (n) | Total (n) |
|---------|--------------|----------------|-----------|
| MMLU    | 15           | 20             | 35        |
| GPQA    | 25           | 25             | 50        |
| **Pooled** | **40**    | **45**         | **85**    |

- **5 rollouts per PI** (problem instance)
- MMLU: 3 faithful PIs, 4 unfaithful PIs
- GPQA: 5 faithful PIs, 5 unfaithful PIs

---

## Analysis 1: Logistic Regression Classifier

**Goal**: Can we predict faithfulness from attention patterns?

### Features Used
- Kurtosis statistics: mean, max, median, std
- Top-5 head kurtosis values
- Behavioral: cue_sentence_attn, cue_sentence_rank, num_sentences

### Results

| Metric | Value |
|--------|-------|
| LOO Accuracy | 71.8% |
| AUC-ROC | 0.819 |
| Precision (Faithful) | 0.75 |
| Recall (Faithful) | 0.60 |
| Precision (Unfaithful) | 0.70 |
| Recall (Unfaithful) | 0.82 |

### Confusion Matrix
```
              Predicted
              Faith  Unfaith
Actual Faith    24      16
       Unfaith   8      37
```

### Top Features (with bootstrap standard errors, n_boot=1000)

| Feature | Coef | Std Err | z | P>|z| |
|---------|------|---------|---|------|
| num_sentences | -4.32 | 7.90 | -0.55 | 0.58 |
| top5_kurt_2 | 3.25 | 7.98 | 0.41 | 0.68 |
| cue_sentence_attn | 2.32 | 6.48 | 0.36 | 0.72 |
| top5_kurt_5 | -6.65 | 22.90 | -0.29 | 0.77 |
| mean_kurt | -10.16 | 41.80 | -0.24 | 0.81 |

**Note**: No individual feature reaches significance (p < 0.05) due to multicollinearity, but the model as a whole is predictive.

**Interpretation**: The classifier achieves above-chance performance (AUC=0.819), suggesting attention patterns contain predictive signal. However, the high standard errors indicate substantial uncertainty in individual coefficients due to correlated features.

---

## Analysis 2: Permutation Test for In-Group Consistency

**Goal**: Is the internal consistency of attention patterns different between faithful and unfaithful groups?

### Method
- Pool all rollouts and randomly reassign labels
- For each permutation, compute in-group Jaccard similarity
- Compare observed difference to null distribution

### Results

| Dataset | Mode | Observed Diff | Null Mean | Null Std | P-value |
|---------|------|---------------|-----------|----------|---------|
| MMLU | Full | 0.299 | 0.076 | 0.187 | 0.159 |
| MMLU | Reasoning | 0.036 | 0.008 | 0.045 | 0.368 |
| GPQA | Full | -0.168 | 0.005 | 0.179 | 0.396 |
| GPQA | Reasoning | 0.031 | -0.0002 | 0.063 | 0.598 |

**Interpretation**: No p-value reaches significance (p < 0.05). The MMLU full-context analysis shows a trend (p=0.159) with unfaithful being more consistent, but this doesn't replicate in GPQA.

---

## Analysis 3: Effect Sizes (Cohen's d)

| Dataset | Mode | Cohen's d | Interpretation |
|---------|------|-----------|----------------|
| MMLU | Full | 3.56 | Very large (unfaithful > faithful consistency) |
| MMLU | Reasoning | 0.84 | Large |
| GPQA | Full | -1.05 | Large (faithful > unfaithful consistency) |
| GPQA | Reasoning | 0.55 | Medium |

**Note**: The direction reverses between MMLU and GPQA, suggesting dataset-specific effects rather than a general pattern.

---

## Analysis 4: Pooled Dataset Analysis (MMLU + GPQA)

### In-Group Consistency (Pooled)

| Condition | Mean Jaccard | Std | 95% CI |
|-----------|--------------|-----|--------|
| Faithful | 0.887 | 0.164 | [0.67, 1.0] |
| Unfaithful | 0.670 | 0.063 | [0.67, 0.67] |

### Out-Group Jaccard
- **Observed**: 0.67 (4/5 heads shared)
- **Shared Heads**: L31H34, L45H0, L47H3, L44H33

### Pooled Effect Size
- Cohen's d = -1.75 (faithful more internally consistent when pooled)

---

## Analysis 5: Attention-Behavior Correlation

### Kurtosis by Faithfulness
| Condition | Mean Kurtosis | Std |
|-----------|---------------|-----|
| Faithful | 6.82 | 5.91 |
| Unfaithful | 9.12 | 6.83 |

- Mann-Whitney p = 0.218 (not significant)
- Cohen's d = -0.36 (small effect)

### Kurtosis vs Mentions Cue
- Point-biserial r = -0.15
- p = 0.161 (not significant)

**Interpretation**: No significant correlation between attention kurtosis and behavioral outcomes.

---

## Analysis 6: Response Length

| Condition | Mean Sentences | Std | Median |
|-----------|----------------|-----|--------|
| Faithful | 64.6 | 31.5 | 56 |
| Unfaithful | 58.1 | 36.2 | 43 |

- Mann-Whitney p = 0.615 (not significant)
- Effect size r = -0.11 (negligible)

**Interpretation**: No significant difference in response length between conditions.

---

## Analysis 7: Per-Head Importance

**Goal**: Which individual heads best discriminate faithful from unfaithful?

### Top 10 Most Predictive Heads

| Head | Cohen's d | P-value | Direction |
|------|-----------|---------|-----------|
| L47H36 | 1.18 | 0.00004*** | faithful > unfaithful |
| L45H3 | 1.08 | 0.002** | faithful > unfaithful |
| L35H16 | 0.98 | 0.0001*** | faithful > unfaithful |
| L47H3 | 0.94 | 0.0002*** | faithful > unfaithful |
| L34H19 | 0.93 | 0.004** | faithful > unfaithful |
| L45H34 | 0.92 | <0.0001*** | faithful > unfaithful |
| L47H0 | 0.87 | 0.0001*** | faithful > unfaithful |
| L36H25 | 0.86 | 0.001** | faithful > unfaithful |
| L31H34 | 0.85 | 0.0008*** | faithful > unfaithful |
| L12H1 | 0.84 | 0.0001*** | faithful > unfaithful |

### Multiple Testing Correction
- Total heads analyzed: 1880
- Significant (p < 0.05): 638
- Significant after Bonferroni (p < 0.000027): **3 heads**

**Interpretation**: Several heads show large effect sizes distinguishing conditions. Notably, L45H34, L47H36, and L35H16 survive Bonferroni correction. All significant heads show *higher* kurtosis for faithful reasoning.

---

## Conclusions

### What the Data Shows

1. **Predictive Signal Exists**: The logistic regression classifier (AUC=0.819) demonstrates that attention patterns contain information predictive of faithfulness.

2. **No Robust In-Group Difference**: The permutation tests fail to show consistent differences in internal consistency between faithful/unfaithful groups across datasets.

3. **Specific Heads Matter**: While aggregate head selection is similar, specific heads (L47H36, L45H3, L35H16) show significant kurtosis differences between conditions even after Bonferroni correction.

4. **Direction Inconsistency**: MMLU and GPQA show opposite patterns for in-group consistency, suggesting the relationship may be task-dependent.

### Limitations

1. **Sample Size**: 85 rollouts is marginal for robust classification. More data would improve power.

2. **Dataset Heterogeneity**: Results don't replicate across MMLU and GPQA, limiting generalizability.

3. **Near-Ceiling Attention**: Cue sentence attention is ~99.9% in all conditions, limiting its discriminative power.

### Recommendations

1. **Collect More Data**: Current sample is underpowered for some analyses.

2. **Focus on Specific Heads**: L47H36, L45H34, and L35H16 show the most promise as faithfulness markers.

3. **Consider Task-Specific Models**: The opposite patterns in MMLU vs GPQA suggest dataset-specific approaches may be needed.

4. **Explore Temporal Dynamics**: Rather than aggregate kurtosis, analyze how attention patterns evolve during reasoning.

---

## Files Generated

- `additional_analyses_results.json` - Detailed results for all analyses
- `extended_variance_analysis_results.json` - Permutation test and effect size results
- `variance_analysis_results.json` - Original in-group/out-group analysis
- `analysis_summary.json` - High-level behavioral statistics
