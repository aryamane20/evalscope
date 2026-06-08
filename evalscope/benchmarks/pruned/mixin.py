"""PruningMixin: content-derived dataset pruning for any DefaultDataAdapter.

Features are extracted from each live Sample at load time (prompt length,
metadata fields, etc.). No shipped prediction or review file is ever read here;
that path belongs exclusively to ``evalscope_ext/tools/offline_analysis.py``.

Pruning works for any model because it only sees the dataset questions, not
model outputs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from evalscope.api.dataset import DatasetDict, Sample


class PruningMixin:
    """Mixin that adds content-derived pruning to any DefaultDataAdapter subclass.

    MRO placement::

        class FooPrunedAdapter(PruningMixin, FooAdapter): ...

    Subclasses must implement ``_extract_raw(sample, subset_name) -> dict`` and
    call ``_init_pruning_params(...)`` from their ``__init__`` after
    ``super().__init__()``.

    ``load_dataset()`` is overridden with a two-pass algorithm:
      1. Load the full (possibly date-filtered) dataset via the parent chain.
      2. Extract content-derived raw measurements from every live Sample.
      3. Compute quantile thresholds globally and build pruning-engine items.
      4. Run the pruning engine to choose ``prune_ratio`` fraction of samples.
      5. Rebuild a DatasetDict containing only the kept samples.
    """

    pruning_strategy_name: str = 'stratified_coreset'
    prune_ratio: float = 0.1
    pruning_seed: int = 42
    n_diff_buckets: int = 5

    def _init_pruning_params(
        self,
        default_strategy: str = 'stratified_coreset',
        default_ratio: float = 0.1,
        default_seed: int = 42,
        default_buckets: int = 5,
    ) -> None:
        """Resolve pruning parameters from both the nested and flat contracts.

        Resolution order for each parameter:
          1. Nested form: ``--dataset-args '{"extra_params":{"prune_ratio":0.1}}'``
          2. Flat form (matches the official spec):
             ``--dataset-args '{"pruning_strategy":"...","prune_ratio":0.1}'``
          3. The benchmark's registered default.

        The flat keys are dropped by ``BenchmarkMeta._update()`` (they are
        neither attributes nor ``extra_params``), so they are recovered here by
        reading the raw ``dataset_args`` retained on the task config.
        """
        ep = self.extra_params or {}

        flat: Dict[str, Any] = {}
        task_config = getattr(self, '_task_config', None)
        if task_config is not None:
            dataset_args = getattr(task_config, 'dataset_args', None) or {}
            flat = dataset_args.get(self.name, {}) or {}
        nested = flat.get('extra_params', {}) or {}

        def pick(default: Any, *keys: str) -> Any:
            for key in keys:
                if key in nested:
                    return nested[key]
            for key in keys:
                if key in flat:
                    return flat[key]
            for key in keys:
                if key in ep:
                    return ep[key]
            return default

        self.pruning_strategy_name = pick(default_strategy, 'pruning_strategy')
        self.prune_ratio = float(pick(default_ratio, 'prune_ratio'))
        self.pruning_seed = int(pick(default_seed, 'seed', 'prune_seed'))
        self.n_diff_buckets = int(pick(default_buckets, 'n_diff_buckets'))

    def load_dataset(self):  # type: ignore[override]
        """Load via parent, then prune by content-derived features."""
        full_dataset_dict = super().load_dataset()  # type: ignore[misc]
        return self._apply_pruning(full_dataset_dict)

    def _apply_pruning(self, dataset_dict):
        """Two-pass: collect, quantize, select, rebuild."""
        from evalscope.api.dataset import DatasetDict as DD, MemoryDataset
        from evalscope.utils.logger import get_logger
        logger = get_logger()

        entries: List[Tuple[str, Any]] = []
        for subset_name, dataset in dataset_dict.items():
            for sample in dataset:
                entries.append((subset_name, sample))

        if not entries:
            logger.warning(f'{self.name}: no samples after parent load; skipping pruning.')  # type: ignore[attr-defined]
            return dataset_dict

        raw_list = [self._extract_raw(s, sn) for sn, s in entries]
        items = self._build_items(raw_list)

        from evalscope.benchmarks.pruned.pruning import get_strategy
        strategy = get_strategy(self.pruning_strategy_name)()
        kept_set: Set[int] = set(
            strategy.select(items, self.prune_ratio, seed=self.pruning_seed)
        )

        n_total = len(entries)
        n_kept = len(kept_set)
        logger.info(
            f'{self.name}: {n_total} -> {n_kept} samples '  # type: ignore[attr-defined]
            f'(strategy={self.pruning_strategy_name}, ratio={self.prune_ratio})'
        )

        bucket_stats: Dict[str, Tuple[int, int]] = {}
        for i, item in enumerate(items):
            b = item['features']['bucket']
            tot, kpt = bucket_stats.get(b, (0, 0))
            bucket_stats[b] = (tot + 1, kpt + (1 if i in kept_set else 0))
        for b, (tot, kpt) in sorted(bucket_stats.items()):
            logger.info(f'  bucket {b!r}: {tot} total -> {kpt} kept')

        pruned_by_subset: Dict[str, list] = {}
        for i, (sn, sample) in enumerate(entries):
            pruned_by_subset.setdefault(sn, [])
            if i in kept_set:
                pruned_by_subset[sn].append(sample)

        self.test_dataset = DD({  # type: ignore[attr-defined]
            k: MemoryDataset(v, name=k) for k, v in pruned_by_subset.items()
        })
        return self.test_dataset

    def _extract_raw(self, sample, subset_name: str) -> Dict[str, Any]:
        """Extract content-derived measurements from a live Sample.

        Returns a dict with:
          ``category``          str   primary stratum label
          ``difficulty_value``  float raw numeric measurement; higher = harder
          ``embedding``         list  optional; enables coreset diversity

        Must not read any score / pass / acc / prediction field.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement _extract_raw')

    def _build_items(self, raw_list: List[Dict[str, Any]]) -> List[dict]:
        """Convert raw measurements to pruning-engine items.

        Quantile thresholds are computed globally from all ``difficulty_value``
        entries so the bucket boundaries adapt to the actual distribution.
        """
        diff_values = [float(r.get('difficulty_value', 0.0)) for r in raw_list]
        sorted_d = sorted(diff_values)
        n = len(sorted_d)
        nb = self.n_diff_buckets

        thresholds = [sorted_d[max(0, int(n * i / nb))] for i in range(1, nb)]

        min_d, max_d = sorted_d[0], sorted_d[-1]
        span = (max_d - min_d) or 1.0

        items = []
        for i, raw in enumerate(raw_list):
            dv = float(raw.get('difficulty_value', 0.0))
            cat = raw.get('category', 'default')

            qidx = sum(dv >= t for t in thresholds)
            bucket = f'{cat}__q{qidx}'

            features: Dict[str, Any] = {
                'bucket': bucket,
                'difficulty': (dv - min_d) / span,
            }
            if 'embedding' in raw:
                features['embedding'] = raw['embedding']

            items.append({'index': i, 'features': features})

        return items
