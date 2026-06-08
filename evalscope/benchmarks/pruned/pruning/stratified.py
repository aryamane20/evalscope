import random
from typing import Dict, List

from .base import PruningStrategy, register_strategy


@register_strategy("stratified")
class StratifiedStrategy(PruningStrategy):
    """Sample proportionally from each category bucket.

    Groups items by ``item["features"][bucket_key]`` and keeps a count from each
    group proportional to its share of the full set, preserving the category
    distribution. Slot allocation uses the largest-remainder method so the total
    is exact.
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

        kept: List[int] = []
        for key, group in groups.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            kept.extend(item["index"] for item in shuffled[: floors[key]])

        return kept
