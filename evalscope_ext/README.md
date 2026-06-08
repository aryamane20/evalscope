# Task 2: Pruned Benchmark Variants

Compact, representative subsets of LiveCodeBench, AA-LCR, and MMMU that give a
trustworthy "is this model good enough?" signal in 10-20% of the evaluation time.
The pruner lives inside evalscope as clean, mergeable code.

## Quick start (mock model, no paid API call)

Demonstrates the full contract using evalscope's built-in `MockLLM`:

```bash
# From repo root
python -m evalscope_ext.tools.mock_contract
```

This:
1. Creates a 30-record synthetic dataset at the evalscope cache location.
2. Runs `live_code_bench` (full, 20 samples) with `MockLLM`.
3. Runs `live_code_bench_pruned` twice (flat contract and nested contract) and
   confirms both keep an identical, reduced sample count (20 loaded, 4 kept).
4. Runs `compare_runs` and confirms the table prints without error.

## Reviewer commands (real model, OpenAI-compatible endpoint)

Substitute `<model>` and `<api-url>` for your deployment.

Pruning parameters are `pruning_strategy`, `prune_ratio`, and `seed` (alias
`prune_seed`). They can be passed two ways. The flat form below matches the
official spec; the nested form is an alternative.

### LiveCodeBench

```bash
# 1. Full eval
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets live_code_bench \
  --dataset-args '{"subset_list":["release_v6"]}' \
  --work-dir ./results_full --no-timestamp

# 2. Pruned eval, FLAT contract (matches the spec): 20% kept, stratified
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets live_code_bench_pruned \
  --dataset-args '{"pruning_strategy":"stratified","prune_ratio":0.2,"subset_list":["release_v6"]}' \
  --work-dir ./results_pruned --no-timestamp

# 3. Compare
python -m evalscope_ext.tools.compare_runs \
  --full ./results_full --pruned ./results_pruned --tolerance 0.05
```

Nested form (alternative) for the same pruned eval:

```bash
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --eval-type openai_api \
  --datasets live_code_bench_pruned \
  --dataset-args '{"extra_params":{"pruning_strategy":"stratified","prune_ratio":0.2},"subset_list":["release_v6"]}' \
  --work-dir ./results_pruned --no-timestamp
```

### AA-LCR

```bash
evalscope eval --model <model> --eval-type openai_api \
  --datasets aa_lcr \
  --work-dir ./results_full --no-timestamp

# Flat contract (matches the spec)
evalscope eval --model <model> --eval-type openai_api \
  --datasets aa_lcr_pruned \
  --dataset-args '{"pruning_strategy":"stratified","prune_ratio":0.2}' \
  --work-dir ./results_pruned --no-timestamp

# Nested contract (alternative)
evalscope eval --model <model> --eval-type openai_api \
  --datasets aa_lcr_pruned \
  --dataset-args '{"extra_params":{"pruning_strategy":"stratified","prune_ratio":0.2}}' \
  --work-dir ./results_pruned --no-timestamp

python -m evalscope_ext.tools.compare_runs \
  --full ./results_full --pruned ./results_pruned --tolerance 0.05
```

### MMMU

```bash
evalscope eval --model <model> --eval-type openai_api \
  --datasets mmmu \
  --work-dir ./results_full --no-timestamp

# Flat contract (matches the spec)
evalscope eval --model <model> --eval-type openai_api \
  --datasets mmmu_pruned \
  --dataset-args '{"pruning_strategy":"stratified","prune_ratio":0.1}' \
  --work-dir ./results_pruned --no-timestamp

# Nested contract (alternative)
evalscope eval --model <model> --eval-type openai_api \
  --datasets mmmu_pruned \
  --dataset-args '{"extra_params":{"pruning_strategy":"stratified","prune_ratio":0.1}}' \
  --work-dir ./results_pruned --no-timestamp

python -m evalscope_ext.tools.compare_runs \
  --full ./results_full --pruned ./results_pruned --tolerance 0.05
```

## Offline evidence (no live model required)

The ratio x strategy sweep validates the pruning strategy against the three
shipped models (gpt-oss-120b, kimi-k2.5, minimax-m2.5) using pre-computed
predictions and reviews:

```bash
python -m evalscope_ext.tools.prune_sweep \
  --evals-dir "/path/to/Evals/Part 1" \
  --ratios 0.10 0.15 0.20 0.25 0.30 \
  --strategies stratified stratified_coreset difficulty_spread \
  --n-seeds 1000
```

Key findings from the sweep:
- `stratified` is the recommended strategy (least biased, most stable).
- LCB: ranking preserved from ratio=0.10 with `stratified`; however, kimi vs
  minimax (1 pp apart) are not reliably separable below ~200 samples. Use ratio
  >= 0.15 for production.
- AA-LCR: LLM judge noise dominates at 10 samples; use ratio >= 0.20.
- MMMU: subject x image-type stratification ensures perception-heavy questions
  (charts, diagrams, medical images, 75% of the 660-sample set) stay represented.

## Architecture

```
evalscope/benchmarks/pruned/
  mixin.py                          PruningMixin: content-derived features, score-free
  pruning/
    base.py                         PruningStrategy ABC + REGISTRY + get_strategy
    stratified.py                   register_strategy("stratified")
    coreset.py                      register_strategy("coreset")
    difficulty.py                   register_strategy("difficulty_spread")
    stratified_coreset.py           register_strategy("stratified_coreset"), default
  live_code_bench_pruned_adapter.py feature: prompt length quantile
  aa_lcr_pruned_adapter.py          feature: input_tokens quantile
  mmmu_pruned_adapter.py            feature: subject x img_type group

evalscope_ext/tools/
  compare_runs.py                   --full <dir> --pruned <dir>, PASS/FAIL table
  offline_analysis.py               review-file based validation (shipped models)
  prune_sweep.py                    ratio x strategy sweep + bootstrap CI
  mock_contract.py                  end-to-end contract demo with MockLLM

tests/
  test_pruning_engine.py            pytest suite, pure Python + numpy
```

## Pruning contract: flat and nested

The mixin resolves each pruning parameter in this order:
1. Nested form: `--dataset-args '{"extra_params":{"prune_ratio":0.1}}'`
2. Flat form (matches the spec): `--dataset-args '{"prune_ratio":0.1}'`
3. The benchmark's registered default.

A reviewer copy-pasting the spec's exact flat command gets real pruning, not
silent defaults.

## Pruning features (all content-derived, never score-derived)

| Benchmark | Feature | Source field |
|---|---|---|
| `live_code_bench_pruned` | Prompt char length quantile | `sample.input[0].content` |
| `aa_lcr_pruned` | Context token count quantile | `sample.metadata['input_tokens']` |
| `mmmu_pruned` | Subject x img_type_group | `subset_name` + `sample.metadata['img_type']` |

Offline analysis uses cross-model avg(total_tokens) from prediction files as a
richer difficulty signal for LCB (longer solutions mean harder problems).
