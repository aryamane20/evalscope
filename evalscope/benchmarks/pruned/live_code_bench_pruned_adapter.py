from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.benchmarks.pruned.mixin import PruningMixin
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

A compact, representative subset of LiveCodeBench produced by the pruning engine.
The subset preserves the question-length distribution of the full benchmark so that
a pass/fail decision on the pruned set matches the full benchmark within tolerance.

## Feature source (live, no shipped files needed)

Buckets are derived at load time from the prompt text length of each question
(chars of ``sample.input[0].content``). Questions are split into
``n_diff_buckets`` equal-count quantile bins; the pruning engine samples
proportionally from each bin, with a slight over-weight to the middle lengths
(difficulty_spread fallback). Date is NOT used as the primary stratum.

## Reviewer commands

    # Full eval
    evalscope eval --model <m> --datasets live_code_bench \\
        --dataset-args '{"subset_list":["release_v6"]}' --output ./results_full/

    # Pruned eval (flat form, matches the spec)
    evalscope eval --model <m> --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy":"stratified_coreset","prune_ratio":0.1, \\
                         "subset_list":["release_v6"]}' \\
        --output ./results_pruned/

    # Compare
    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ --pruned ./results_pruned/ --tolerance 0.05

    # Offline evidence (uses shipped review files, NOT needed for the live eval)
    python -m evalscope_ext.tools.offline_analysis \\
        --evals-dir ./shipped_evals/ --benchmark live_code_bench \\
        --strategy stratified_coreset --prune-ratio 0.1
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=[
            'release_v6',
            'release_latest',
            'release_v1',
            'release_v2',
            'release_v3',
            'release_v4',
            'release_v5',
            'v1',
            'v1_v2',
            'v1_v3',
            'v1_v4',
            'v1_v5',
            'v1_v6',
            'v2',
            'v2_v3',
            'v2_v4',
            'v2_v5',
            'v2_v6',
            'v3',
            'v3_v4',
            'v3_v5',
            'v3_v6',
            'v4',
            'v4_v5',
            'v4_v6',
            'v5',
            'v5_v6',
            'v6',
        ],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=
        '### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy name.',
                'value': 'stratified_coreset',
                'choices': ['stratified', 'coreset', 'difficulty_spread', 'stratified_coreset'],
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to KEEP (0 < prune_ratio <= 1).',
                'value': 0.1,
            },
            'seed': {
                'type': 'int',
                'description': 'Random seed for deterministic subset selection.',
                'value': 42,
            },
            'n_diff_buckets': {
                'type': 'int',
                'description': 'Number of quantile buckets for prompt-length stratification.',
                'value': 5,
            },
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems starting from this date (YYYY-MM-DD).',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD).',
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {'shell_executor': {}, 'python_executor': {}},
        },
    )
)
class LiveCodeBenchPrunedAdapter(PruningMixin, LiveCodeBenchAdapter):
    """LiveCodeBench with content-length-stratified pruning.

    Inherits all scoring/inference logic from LiveCodeBenchAdapter.
    PruningMixin.load_dataset() runs first, calls the parent chain for full
    loading (including date-based sample_filter), then prunes by prompt length.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_pruning_params(
            default_strategy='stratified_coreset',
            default_ratio=0.1,
            default_seed=42,
            default_buckets=5,
        )

    def _extract_raw(self, sample, subset_name: str) -> Dict[str, Any]:
        """Bucket by prompt text length (content-derived, not date).

        Field used:  ``sample.input[0].content``  (str, full formatted prompt)
        Category:    ``'lcb'``
        Difficulty:  raw char count, quantile-normalised by PruningMixin
        """
        content = sample.input[0].content if sample.input else ''
        prompt_len = float(len(content)) if isinstance(content, str) else 0.0
        return {
            'category': 'lcb',
            'difficulty_value': prompt_len,
        }
