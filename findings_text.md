# Do Faithful & Unfaithful CoT Differ in How They Focus Attention?

Findings of the Thought Anchors faithfulness project, last update Aug 2026.
Setup: a wrong "Stanford professor thinks the answer is (X)" hint is added to hard multiple-choice
questions. Some rollouts follow the hint and mention it (verbalizing); some follow it without
mentioning it (silent). Model: DeepSeek-R1-Distill-Qwen-14B; datasets: MMLU(-Pro) and GPQA variants.

## Setup and problem selection

- 1,494 problems screened (GPQA main 448, GPQA extended 546, MMLU-Pro 500), each with a wrong professor hint.
- Per problem: 32 hinted rollouts, 20 unhinted, 8 no-reasoning answers (temp 0.7).
- Measured per problem: hint influence (gap), no-reasoning accuracy, reasoning gain, counts of verbalizing (V) and silent (S) cue-following rollouts.
- Analysis set: 36 problems total, on average 26 usable rollouts per problem (the number varies with how many silent and verbalizing rollouts each problem produced). 24 problems where the hint strongly changes answers (gap >= 0.7, with >=3 verbalizing and >=3 silent rollouts), plus 21 meeting stricter criteria (hint influence >= 0.5, no-reasoning accuracy < 0.5, accuracy gain from reasoning >= 0.2, >=3 verbalizing and >=3 silent rollouts); 9 problems are in both groups.
- Rollouts that hit the length limit were resumed at 16k tokens. Average rollout length: 219 sentences on GPQA main, 247 on GPQA extended, 80 on MMLU-Pro (~5.0k / 5.3k / 1.6k tokens).

## 1. Attention statistics do not separate silent from verbalizing rollouts

- Hypothesis: silent hint-following has a recognizable attention signature in terms of "thought anchors".
- First version compared consistently-silent problems against consistently-verbalizing ones (p = 0.04). But upon further investigation, the same kurtosis gap shows up between those problem groups in uncued rollouts, where no silent/verbalizing behavior exists. So it is not about faithfulness, just problem selection.
- Fix: compare within problems - study the ones where the same problem produces both silent and verbalizing rollouts, each compared with its own.
- Length becomes a confounder. Uncued rollouts of these problems run ~200 sentences; verbalizing rollouts ~192 (statistically same as uncued); silent ones ~186 (significantly shorter, p = 0.001). So silent following shortens deliberation - the hint substitutes for reasoning - while verbalizers deliberate like hint-free rollouts and mention the hint late (~65% through, then continue). The mention text itself is tiny, and 41% of same-problem pairs have the verbalizing one being shorter: the behaviors are mildly shifted length distributions, not mechanical add-ons. Kurtosis tracks length at correlation 0.98, which amplifies this mild shift into an apparent attention difference. Within-problem on an earlier 76-problem collection: raw differences 74% positive (p < 0.001); at similar lengths: 46%, p = 0.53 - the effect is gone.
- On the main dataset described above (36 problems, ~26 rollouts each): same result. Entropy null everywhere.
- How the score works: each head gets a kurtosis over how much later sentences look back at each sentence (high = look-back concentrated on a few sentences); a rollout's score is the mean of its top-20 heads - the receiver heads. Thought Anchors (Bogdan et al., arXiv:2506.19143, Section 4.1) defines these as "attention heads that narrow attention toward specific sentences", scored by "the mean of [each sentence's] columns below the diagonal to measure the extent each sentence receives attention from all downstream sentences". The paper's method leaves these out: attention from the few sentences immediately after (the model always reads what it just wrote, so that says nothing), and the opening sentences (the sequence start attracts attention for its position, not its content). Rank normalization is applied but it help here: it rescales values within each row, but the sink is still the top-ranked target of nearly every row, so it would still dominate; dropping it is the only way to remove it. The table shows the verbalizing-minus-silent difference d (standard-deviation units, paired within problem, 21-problem strict sample) under different settings:
metric versionrawlength-controlledlighter exclusion windows (skip 1 sentence, ignore 3 neighbors)d = +0.79, p = 0.0007d = +0.33, p = 0.22paper's exclusion windows (skip 10, ignore 20, reasoning sentences only)d = +0.71, p = 0.007d = +0.39, p = 0.29rank-normalized (paper's depth control)d = +0.37, p = 0.09d = +0.02, p = 0.84
Same verdict in every version: the raw difference is significant but carried by rollout length; controlled for length, nothing remains. Rank normalization alone already halves the length coupling and most of the "signal" with it.

[FIGURE 1: paired differences]

## 2. Hint position changes following and mentioning

- Hint at prompt start: following often silent (37% of followers verbalize).
- Hint mid-prompt: following halves; ~90% of followers verbalize.
- Among followers, mentioning tracks the hint's decisiveness: the more the hint moves a problem's answers, the more often followers mention it (correlation +0.24, p = 3e-7, 457 problems); silent following concentrates where the hint nudges rather than dominates.

## 3. Hint influence and reasoning ability work against each other

- Across 1,493 problems, hint following is inversely related to both kinds of accuracy, but much more strongly to accuracy with reasoning (correlation -0.49) than to accuracy answering instantly without reasoning (-0.18). Hint influence (following minus the no-hint baseline) shows the same pattern (-0.37 vs -0.14).
- Parallel with Boppana et al. (arXiv:2603.05488): they find the CoT role depends on difficulty - on easy problems the model commits to its answer internally at the start and the CoT text is performative; on hard problems belief and text develop together. The result here is complementary: where reasoning genuinely works, hints do not sway the answer; where reasoning fails, hints are what the model relies on.

[FIGURE 2: Hint following vs accuracy with and without reasoning]

## 4. We can't predict upcoming verbalization from the model's activations

- Design from Boppana et al.: a linear classifier reads the state at each sentence and predicts whether the rollout will mention the hint or stay silent. Positions at or after a mention are excluded; there is a held-out test set. Chance-level (AUC 0.50-0.55) through the first 60% of the reasoning, at most 0.63 near the end. So verbalization is not really pre-committed - unlike the upcoming answer in Boppana et al.

[FIGURE 3: Probe AUC along the reasoning]

---
Repo: github.com/yulia-volkova/thought-anchors-faithfulness
