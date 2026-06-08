import math
import random
from typing import List

from .base import PruningStrategy, register_strategy


@register_strategy("difficulty_spread")
class DifficultySpreadStrategy(PruningStrategy):
    """Keep a spread across the difficulty range, with a slight middle bias.

    Items are sorted by ``item["features"]["difficulty"]`` and divided into
    equal-width quantile buckets. Slot allocation favours the middle buckets via
    a Gaussian-shaped weight so neither purely easy nor purely hard items
    dominate. This is not a top-k by difficulty.
    """

    def __init__(self, n_buckets: int = 5):
        self.n_buckets = n_buckets

    def select(self, items: List[dict], prune_ratio: float, seed: int = 0) -> List[int]:
        if not items:
            return []
        if prune_ratio >= 1.0:
            return [item["index"] for item in items]

        rng = random.Random(seed)
        n = len(items)
        k = max(1, round(n * prune_ratio))
        n_buckets = min(self.n_buckets, n, k)

        sorted_items = sorted(
            items, key=lambda x: x["features"].get("difficulty", 0.5)
        )
        buckets: List[List[dict]] = [[] for _ in range(n_buckets)]
        for rank, item in enumerate(sorted_items):
            bucket_id = min(int(rank * n_buckets / n), n_buckets - 1)
            buckets[bucket_id].append(item)

        middle = (n_buckets - 1) / 2.0
        spread = max(n_buckets / 4.0, 1.0)
        weights = [
            1.0 + 0.5 * math.exp(-0.5 * ((i - middle) / spread) ** 2)
            for i in range(n_buckets)
        ]
        total_w = sum(weights)

        exact = [w / total_w * k for w in weights]
        floors = [int(c) for c in exact]
        remainders = [exact[i] - floors[i] for i in range(n_buckets)]
        remaining = k - sum(floors)
        for i in sorted(range(n_buckets), key=lambda x: -remainders[x])[:remaining]:
            floors[i] += 1

        kept: List[int] = []
        for bucket, n_keep in zip(buckets, floors):
            n_keep = min(n_keep, len(bucket))
            shuffled = list(bucket)
            rng.shuffle(shuffled)
            kept.extend(item["index"] for item in shuffled[:n_keep])

        return kept
