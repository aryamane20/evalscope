from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter, PROMPT_TEMPLATE
from evalscope.benchmarks.pruned.mixin import PruningMixin
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

A compact, representative subset of AA-LCR (Artificial Analysis Long Context Retrieval)
produced by the pruning engine. The subset preserves the context-length distribution
of the full benchmark.

## Feature source (live, no shipped files needed)

Buckets are derived at load time from ``sample.metadata['input_tokens']``, the
pre-computed token count of the full long-context prompt. This is the primary
difficulty signal: longer contexts are harder. Questions are split into
``n_diff_buckets`` equal-count quantile bins and sampled proportionally.

No shipped prediction or review file is read at eval time.

## Reviewer commands

    # Full eval
    evalscope eval --model <m> --datasets aa_lcr --output ./results_full/

    # Pruned eval (flat form, matches the spec)
    evalscope eval --model <m> --datasets aa_lcr_pruned \\
        --dataset-args '{"pruning_strategy":"stratified_coreset","prune_ratio":0.1}' \\
        --output ./results_pruned/

    # Compare
    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ --pruned ./results_pruned/ --tolerance 0.05
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
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
                'description': 'Number of quantile buckets for context-length stratification.',
                'value': 5,
            },
            'text_dir': {
                'type': 'str | null',
                'description': 'Local dir with extracted AA-LCR text files; null = auto-download.',
                'value': None,
            },
        },
    )
)
class AALCRPrunedAdapter(PruningMixin, AALCRAdapter):
    """AA-LCR with context-length-stratified pruning.

    Inherits auto-download, document loading, and LLM-judge scoring from
    AALCRAdapter. PruningMixin.load_dataset() calls the parent chain for full
    loading (including document download), then prunes by context length.
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
        """Bucket by context length from sample.metadata['input_tokens'].

        Primary field:  ``sample.metadata['input_tokens']``  (int, token count)
        Fallback field: ``len(sample.input[0].content)``     (char count)
        Category:       ``'aalcr'``
        Difficulty:     raw token count, quantile-normalised by PruningMixin
        """
        metadata = sample.metadata or {}
        input_tokens = float(metadata.get('input_tokens', 0) or 0)

        if not input_tokens:
            content = sample.input[0].content if sample.input else ''
            input_tokens = float(len(content)) if isinstance(content, str) else 0.0

        return {
            'category': 'aalcr',
            'difficulty_value': input_tokens,
        }
