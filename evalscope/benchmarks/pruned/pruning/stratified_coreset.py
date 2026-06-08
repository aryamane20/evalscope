import random
from typing import Dict, List

from .base import PruningStrategy, register_strategy


@register_strategy("stratified_coreset")
class StratifiedCoresetStrategy(PruningStrategy):
    """Default strategy: stratify by category, then diversify within each stratum.

    1. Groups items by ``item["features"][bucket_key]`` and allocates slots
       proportionally (largest-remainder, same as StratifiedStrategy).
    2. Within each group, selects items via coreset diversity if embeddings are
       present, or difficulty_spread (full range coverage) as the fallback,
       which also serves as the tie-breaker when embeddings are absent.
    """

    def __init__(self, bucket_key: str = "bucket"):
        self.bucket_key = bucket_key

    def select(self, items: List[dict], prune_ratio: float, seed: int = 0) -> List[int]:
        if not items:
            return []
        if prune_ratio >= 1.0:
            return [item["index"] for item in items]

        rng = random.Random(seed)
        n = len(items)
        total_keep = max(1, round(n * prune_ratio))

        groups: Dict[str, List[dict]] = {}
        for item in items:
            key = item["features"].get(self.bucket_key, "default")
            groups.setdefault(key, []).append(item)

        exact: Dict[str, float] = {k: len(v) / n * total_keep for k, v in groups.items()}
        floors: Dict[str, int] = {k: int(v) for k, v in exact.items()}
        remainders: Dict[str, float] = {k: exact[k] - floors[k] for k in groups}
        remaining = total_keep - sum(floors.values())
        for k in sorted(remainders, key=lambda x: -remainders[x])[:remaining]:
            floors[k] += 1

        has_embeddings = all("embedding" in item["features"] for item in items)

        from .coreset import CoresetStrategy
        from .difficulty import DifficultySpreadStrategy

        kept: List[int] = []
        for bucket_key, group in groups.items():
            n_keep = floors[bucket_key]
            if n_keep <= 0 or not group:
                continue

            group_seed = rng.randint(0, 2**31 - 1)
            sub_ratio = n_keep / len(group)

            if has_embeddings:
                sub = CoresetStrategy()
            else:
                sub = DifficultySpreadStrategy()

            kept.extend(sub.select(group, sub_ratio, seed=group_seed))

        return kept
