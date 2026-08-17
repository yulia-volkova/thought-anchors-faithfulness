# Attention Patterns and Chain-of-Thought Faithfulness: A Controlled Investigation

Yulia Volkova - Athena Fellowship / MATS 9.1 extension - Nov 2025 to Aug 2026.
Draft for review.
Repository (reviewers: please inspect the code and result files directly):
https://github.com/yulia-volkova/thought-anchors-faithfulness
(key files: FEEDBACK_AND_PLAN.md, extract_within_problem.py, entropy_analysis.py, hiring_attention.py, cue_conditioning_control.py)

> **Review prompt (for the reviewing agent):** You are reviewing a research
> write-up on attention patterns and chain-of-thought faithfulness. Be
> adversarial and specific; cite section numbers. Priorities:
> 1. Attack the causal reading of the hiring result (Section 5): the
>    attention-sensitivity correlation is correlational - list confounds not
>    yet controlled (beyond decision confidence and length) and how each
>    would produce rho=-0.64 without covert use of the attribute.
> 2. Audit every claim for slippage from "verbalization" to "faithfulness"
>    (Section 6.1 explains why they are not the same).
> 3. Scrutinize the length-matched kurtosis concordance of 0.605 over 215
>    non-independent pairs (Section 4): is calling it "a weak residual
>    trend" too strong, too weak, or right?
> 4. Check statistical hygiene: paired tests at problem level, GroupKFold,
>    best-layer selection (is reporting best-layer AUC without nesting
>    acceptable as framed?).
> 5. Say what one additional experiment would most change your confidence
>    in each of the three hypothesis verdicts.
> Format: numbered findings, each with severity (blocking / major / minor),
> the exact quoted claim, and a suggested fix.


## Abstract

We test whether unsupervised attention statistics (receiver-head kurtosis;
normalized attention entropy) distinguish chain-of-thought rollouts that
verbalize a hint they follow from rollouts that follow it silently. On
DeepSeek-R1-Distill-Qwen-14B over MMLU and GPQA-Diamond, with problem identity
held fixed and analysis restricted to problems where the hint is demonstrably
causal (cue-response gap >= 0.5) and direct-answer accuracy is below 0.5, we find no usable unsupervised signal: apparent
effects are explained by a verbalization-length coupling (verbalizing rollouts
are ~20 sentences longer; kurtosis correlates rho=0.98 with length). Supervised activation probes on the same forward passes reach AUC 0.86 post-CoT (best-layer, exploratory; pre-CoT probes retracted pending re-extraction), and attention statistics add nothing on top. In a separate
decision setting (resume screening, Gemma-3-12B), targeted attention from the
decision token to a candidate's name is strongly INVERSELY related to the
name's causal effect on the decision (rho=-0.64, p=4e-14): lower final-query attention to candidate names is strongly associated with larger counterfactual name effects at the resume level. This association is exploratory: it does not yet identify covert demographic use, distinguish absorption from attention competition, or establish a per-response detector.

## 1. Hypotheses

- H1 (original, Nov 2025): faithful CoT (verbalizes the hint it follows)
  anchors attention on key sentences -> higher receiver-head kurtosis;
  unfaithful CoT (follows silently; post-hoc rationalization) shows diffuse
  attention. NOT SUPPORTED after length controls; no consistent cross-dataset residual effect (MMLU and GPQA point in opposite directions; see Sec. 4).
- H2 (diffuseness, Aug 2026): silent rollouts have higher normalized attention
  entropy than verbalizing rollouts of the same problem. NOT SUPPORTED; no consistent residual or matched-pair difference in this setting (no equivalence bound established yet - powered rerun in progress).
- H3 (targeted attention, hiring): attention from the decision token to a
  demographic attribute predicts that attribute's causal effect on the
  decision. STRONG INVERSE EXPLORATORY ASSOCIATION (sign opposite to prediction); mechanism and predictive generality untested.

## 2. Setting

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B. Datasets: MMLU (143
problems), GPQA-Diamond (186), 20 cued rollouts each; cue = "A Stanford
professor thinks the answer is (X)", X != ground truth. Labels: verbalizing
(follows cue AND mentions it) vs silent (follows without mention) -
terminology per Zaman & Srivastava's critique (see 6.1); we do not equate
silence with unfaithfulness. Attention: sentence-level received-attention
(vert) scores per head (proximity_ignore=3, drop_first=1), Thought Anchors
framework (Bogdan et al. 2025, [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)).

Final analysis population ("strict"): problems with cue-response gap >= 0.5,
no-reasoning accuracy < 0.5, and mixed behavior (>=2 verbalizing and >=2
silent rollouts): 25 problems (15 MMLU, 10 GPQA), 383 rollouts (154 V, 229 S),
~15 rollouts/problem. A wider pool (113 problems / 1,556 rollouts) gives the
same conclusions; numbers below are strict unless noted.

## 3. Controls that dismantled the original result

1. Rollout-level pseudo-replication: the original GPQA p=0.04 disappears at
   problem level (best p=0.056).
2. Cue-conditioning (suggested by Daria & Riya): the cue sentence (position 0)
   was excluded from the metric by drop_first, so direct cue salience cannot
   explain kurtosis; but problems labeled faithful vs unfaithful by cued
   behavior differ in kurtosis even in their UNCUED rollouts (GPQA d~1.2),
   where no (un)faithful behavior exists -> group gaps reflect problem
   selection. Design flaw noted: cue at position 0 coincides with the
   attention sink; successors should place it mid-prompt.
3. Cue-restatement ablation: deleting cue-mentioning sentences from the
   per-head score lists changes kurtosis <1%; cue mentions rank low in
   attention; the dominant spike is typically the first reasoning sentence.

## 4. The decisive within-problem experiment

Design: compare verbalizing vs silent rollouts of the SAME problem
(teacher-forced re-extraction on one H100; custom attention kernel pools
token attention to sentence level in 8-head chunks inside the attention
computation, making 14B x 8k-token extraction feasible; same passes store
activations at 16 layers x 2 positions).

Strict-subset results:
- Raw paired kurtosis (V-S): d=+0.61, p=0.002, 76% of problems positive.
- Verbalizing rollouts are +20.9 sentences longer (p=0.0006); kurtosis
  correlates rho=0.98 with length across rollouts.
- Length-residualized paired test, pooled: d=+0.14, p=0.19; PER DATASET the residual effects point in OPPOSITE directions (MMLU d=+0.86, p=0.002; GPQA d=-0.52, p=0.16) - the pooled null partly reflects cancellation, so the defensible claim is no CONSISTENT cross-dataset residual effect, not that length explains everything. Length-matched pairs (|dn|<=5), exploratory: pair-weighted concordance 0.605 over 215 dependent pairs from 21 problems; problem-weighted 0.551; problem-level Wilcoxon p=0.23 - not distinguishable from chance.
- Entropy (H2), per-query normalized (the rollout-level aggregate is NOT length-invariant) - rho=0.49-0.72 with
  length in practice: raw sign OPPOSITE to H2 (verbalizing more diffuse,
  p=0.03); residualized d~0.0 (p=0.73-0.75); matched pairs 0.488-0.516.
- Supervised probes (GroupKFold by problem; best-layer values are EXPLORATORY, not nested): post-CoT AUC 0.862. Pre-CoT results are RETRACTED pending re-extraction: the original boundary-token derivation was contaminated by the continuation (joint tokenization; fixed in extract_within_problem.py with a same-prompt equality assertion). Kurtosis within-problem AUC 0.64; adding kurtosis to
  the probe does not improve it (0.83 -> 0.83 on the 76-problem set).

Conclusion: no unsupervised global attention signature of unverbalized cue
use; all such statistics inherit the verbalization-length coupling. Post-generation activations predict the keyword-detected cue-mention label (note: post-CoT activations have access to the text that defines the label); pre-generation claims await the repaired extraction.

## 5. Hiring setting: the inverted targeted-attention result

Data: Karvonen & Marks resume-screening logs ([arXiv:2506.10922](https://arxiv.org/abs/2506.10922);
[github.com/adamkarvonen/llm_bias](https://github.com/adamkarvonen/llm_bias); HF adamkarvonen/bias_eval). Model:
google/gemma-3-12b-it (their logged runs, our re-extraction). 111 resumes x 4
name variants (White/Black x Female/Male; each variant swaps a name+pronouns+email
BUNDLE, so race, gender, and lexical name properties are not separable) x 4
prompt versions. Ground truth:
name-sensitivity = the range of p(yes) across the 4 name variants computed WITHIN each prompt version, averaged over versions (isolates the name intervention from prompt-version effects; see hiring_analysis.py). Measure: attention from the final
(decision) token to the name's token span, top-5 heads over the 8 global
attention layers (local layers cannot reach the name).

Scope note: we use ONE condition of their much larger matrix (job-description
screening, "gm_high_bar" prompts v1-v4). Their headline manipulations -
company identity (Meta/Palantir), selective anti-bias statements, locations,
college names - are not used here. Their CoT conditions exist only for closed
API models, so the open-weight logs we can extract attention from are direct
Yes/No decisions without CoT; their paper separately shows CoTs in this
setting look clean while decisions are biased, which is what makes the
setting canonical for unverbalized influence.

Result: attention-to-name is inversely related to name-sensitivity:
resume-level Spearman -0.630 (p=1.2e-13, n=111); -0.562 controlling mean p(yes); length not a confound (rho=-0.04). Exploratory resume-level AUC 0.859 for flagging name-sensitive resumes (pre-specified threshold: mean within-version range > 0.05; 36 sensitive / 75 not). Low final-query attention to the name is associated with larger counterfactual name effects at the resume level.
Interpretations to distinguish: (a) covert absorption - the attribute is
integrated into resume representations during prefill, so the decision token
needn't re-attend; (b) learned suppression of overt attention to sensitive
attributes. Practical reading: low overt attention to a sensitive attribute
plus a borderline decision flags elevated risk of covert influence.

### 5.1 Position profile: the deficit is decision-time only (2026-08-16)

Measuring attention to the name span from every query position (per global
layer, per head; hiring_attention2.py) and correlating with name-sensitivity
at reading positions: early rho=-0.155 (n.s.), mid -0.038, late -0.006;
decision position -0.562 (p<1e-4). The model reads the name normally; only
the final decision position under-consults it. This rules out active
"look-away"/suppression during reading and leaves (a) prefill absorption and
(b) attention competition at the decision token (borderline decisions spread
attention over qualifications) as live interpretations.

### 5.2 College condition: inverse effect for an inferred demographic (2026-08-17)

Their college-affiliation condition signals race indirectly (Howard
University / Morehouse College vs Georgetown / Emory; v3 meta job
description, 480 records = 111 usable groups x 4 college variants, 36
records lack yes_probs). Same measurement with span = the college mention
(hiring_attention3.py, SPAN_MODE=college): resume-level Spearman
(attention-to-college, college-sensitivity) = -0.435 (p=1.8e-6, n=111);
-0.347 controlling mean p(yes); exploratory AUC 0.660 (23 sensitive / 88
not). Weaker than the name bundle but same direction and significant: the
inverse relation does not require the literal name tokens; it holds when
the demographic must be inferred from a credential. Direction note: in this
condition HBCU variants receive higher mean p(yes) (0.444 vs 0.381).

### 5.3 CoT generations on open weights (2026-08-17)

Their CoT results exist only for closed API models. We generated new
decisions with the authors' canonical CoT format ("Chain of thought:
<1-2 sentences> / Answer: <Yes or No>"; yes_no_high_bar_cot.txt) on
gemma-3-12b-it, 8 samples per record at temperature 0.7, for both the name
(444 records) and college (480 records) conditions (hiring_cot_generate.py,
hiring_cot_analysis.py). Findings:

- Name bias survives reasoning, attenuated ~5x: within-resume carryover of
  the direct-mode p(yes) shift into CoT-mode yes-rates has slope +0.179
  (within-group Spearman +0.132, p=5e-3, n=444). Range comparison agrees:
  direct bias range 0.122 vs CoT excess-over-sampling-noise 0.050.
- College bias is largely eliminated by reasoning: CoT excess range +0.006,
  carryover slope +0.015 (weak rank carryover +0.156, p=1e-3).
- Verbalization is absent: 0 of 7,392 CoTs contain an explicit demographic
  word (the only regex hits were "Six Sigma Black Belt" and "key man
  insurance"); the candidate name is restated in 1.4% (names cond); the
  college is mentioned in 0.44% of college-condition CoTs, and on
  college-sensitive resumes 0.41%. String matching is an upper bound on
  citing the attribute as a factor, which strengthens the zero.
- Position vs admissibility: the demographic cue sits mid-prompt - the
  configuration that made hint-following almost fully verbalized in the
  GPQA mid-prompt experiment - yet following is completely silent here. So
  verbalization tracks whether the cue is an admissible reason, not its
  position; the mid-prompt "verbalization fix" does not transfer to
  socially inadmissible cues.

## 6. Relation to prior and concurrent work

1. Zaman & Srivastava, ACL 2026 ([arXiv:2512.23032](https://arxiv.org/abs/2512.23032)): hint-verbalization is a
   flawed faithfulness label; non-verbalized hints causally mediate through
   the CoT (CMA); verbalization rises with sampling budget (faithful@k). We
   adopt their terminology; our +20-sentence coupling is consistent with
   their incompleteness account on a reasoning distill; our MMLU
   max_tokens=2048 sits in their flagged tight-budget regime (limitation).
2. Rationalization probes ([arXiv:2603.17199](https://arxiv.org/abs/2603.17199)): supervised activation probes
   detect motivated reasoning (AUC 65-82%, incl. pre-CoT). We mirror their
   two-position design; our pre-CoT number is retracted pending re-extraction
   (boundary-token bug), so we do not currently claim a replication of the
   pre-generation effect
   on an untested model family; they test no unsupervised or attention-based
   detector - we fill that cell with a negative.
3. Arcuschin et al. ([arXiv:2602.10117](https://arxiv.org/abs/2602.10117)): black-box unverbalized-bias discovery;
   population-level, ~766-2,493 query pairs (~$100) per concept, no
   single-response attribution. Our hiring result is the per-response,
   white-box complement.
4. Thought Anchors (Bogdan et al., [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)) and Thought Branches:
   source of the receiver-head framework; our results caution against
   faithfulness applications of global receiver-head statistics.
5. Karvonen & Marks ([arXiv:2506.10922](https://arxiv.org/abs/2506.10922)): setting, data, and models for
   Section 5.

## 7. Limitations

- CoT experiments: single model family (R1-Distill-Qwen-14B); cue always at
  position 0 (attention sink); MMLU generation budget 2048 tokens; labels are
  verbalization, not ground-truth faithfulness; strict subset is 25 problems
  (paired power ~d>=0.6).
- Hiring: single model (Gemma-3-12B); name-bundle and college conditions,
  one prompt version each for the follow-ups (5.1-5.3); correlational;
  reading-position profile rules out look-away but absorption vs
  decision-time competition remains undistinguished; CoT decisions are
  8 samples/record at temperature 0.7 (rates are coarse; handled via noise
  floor and slope estimands); production inference does not materialize
  attention (FlashAttention), limiting direct deployment of any
  attention-based monitor.

## 8. Open questions

1. Absorption vs suppression: mid-forward name-swap / attention-blocking
   ablation in the hiring setting.
2. Does the inverted attention-sensitivity relation replicate on Gemma-3-27B,
   Mistral-Small-24B (data in hand), and non-Gemma families?
3. Is there any length-invariant attention statistic with signal? (Our
   matched-pair kurtosis trend, 0.605, is the only surviving hint.)
4. Redesigned CoT experiment: cue mid-prompt, generous budgets, second model
   family - does anything change?
5. Can pre-CoT probes + targeted attention combine into a practical
   risk monitor (per-response, cheap)?
6. Latent reasoning models (Uzay's suggestion): do these questions transfer
   when there is no verbalized CoT at all?
7. THE BRIDGE EXPERIMENT (full follow-up design): CoT hiring on an open
   model. Generate step-by-step reasoning rollouts with Gemma-3-12B over the
   Karvonen resumes (~100 resumes x 4 name variants x several rollouts, one
   GPU session; extraction and analysis code in this repo run unchanged).
   Eligibility criterion specific to this setting (not the MCQ criteria):
   resumes where the name-bundle effect SURVIVES reasoning (delta p under CoT
   vs direct, per resume); Karvonen & Marks report bias shrinks but persists
   under CoT and is never verbalized. Their context manipulations (company
   identity, anti-bias statements) can serve as a bias-strength dial to raise
   the yield of sensitive resumes. Two labels per rollout: (a) verbalizing vs silent - does the CoT mention
   the name/demographics; (b) flipped vs not - did the name causally move
   the decision (delta p across name variants; behavioral, judge-free).
   Metrics, in order of priority: targeted attention to the name span and to
   name-restating CoT sentences (primary - the only attention quantity with
   demonstrated signal); sentence-level vert scores with kurtosis/entropy as
   secondary metrics ONLY under length controls (within-resume comparisons,
   matched pairs, residualization); supervised probes from the same forward
   passes as the ceiling. Key new question: does the inverted
   attention-avoidance pattern persist during reasoning, or does CoT change
   where attention goes? Design fixes inherited: cue (name) mid-prompt,
   within-pair comparisons, generous token budget, problem-level statistics.
8. Where does attention reallocate when the model avoids the sensitive
   attribute - diffuse, or re-concentrated on qualification sentences
   (plausible-deniability attention)? Requires saving full decision-token
   attention rows (5-line change to hiring_attention.py).

### Pre-registered addition (2026-08-14): attention-blocking causal test

After the powered rerun's eligibility gate, on eligible problems: generate
cued rollouts with attention to the cue-sentence tokens blocked (weights
zeroed, rows renormalized, all layers, all positions after the cue) and
compare cue-follow rates across cued-normal / cued-blocked / uncued.
Interpretation fixed in advance: blocked ~ uncued floor -> the cue's influence
is mediated by direct attention to it (causal validation of attention-based
monitoring); blocked ~ cued-normal -> influence is absorbed into early-layer
representations and travels without re-attending (the absorption account,
converging with the hiring result). This is the causal leg suggested by
Goodfire in Feb 2026 and previously untested.

## Acknowledgements

Feedback: Adam Karvonen; Jack Merullo & Siddharth Boppana (Goodfire); Uzay
Macar; Daria & Riya; Arun Jose (Athena mentor); Skylar Shibayama & Maria M
(MATS RM feedback on presentation). Compute: user's H100.


## Revision note (2026-08-14)

This draft incorporates an external adversarial review: corrected hiring
estimand (within-version name contrasts), associational language for the
hiring result, pre-CoT probe retraction pending repaired extraction,
clustered matched-pair reporting, per-dataset residual effects, exploratory
labels on best-layer AUCs, and downgraded hypothesis verdicts. In progress:
a pre-registered powered rerun (screening ~1,500 fresh problems; criteria
gap>=0.5, nr accuracy<0.5, reasoning gain>=0.2, >=3V/>=3S; Wilcoxon
alpha=0.01 with d=0.3 equivalence bound) and the pre-CoT re-extraction.
