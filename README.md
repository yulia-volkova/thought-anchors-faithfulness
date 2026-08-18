# Thought Anchors and CoT Faithfulness

Do faithful and unfaithful chains of thought differ in how they focus attention?
A replication/extension of the Thought Anchors framework
([Bogdan et al., arXiv:2506.19143](https://arxiv.org/abs/2506.19143)).

**Most up-to-date results and conclusions:
[thought-anchors-faithfulness.vercel.app](https://thought-anchors-faithfulness.vercel.app/)**
(the Aug 2026 update at the top; the interactive attention explorer below it).

- `dataset_prep/` - building the MMLU/GPQA question sets
- `data_generation/` - generating cued/uncued/no-reasoning rollouts (vLLM)
- `whitebox_extraction/` - recording attention and activations from existing rollouts
- `analysis/` - the statistics behind the findings
- `blocking/` - the attention-blocking causal test
- `probes/` - trajectory probes (predicting upcoming verbalization from activations)
- `hiring/` - side experiments on the Karvonen & Marks resume-screening data
- `original_pipeline/` - the January 2026 pipeline (superseded; kept for the record)
- `webapp/` - the results website

Run scripts from the repo root (shared modules `anchors_utils.py`, `hf_utils.py`,
`select_mixed_problems.py` live here).
