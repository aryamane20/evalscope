from typing import List

import numpy as np

from .base import PruningStrategy, register_strategy


@register_strategy("coreset")
class CoresetStrategy(PruningStrategy):
    """Diversity selection via farthest-point (k-center greedy) sampling.

    Each successive point is the one maximally far from the already-chosen set,
    guaranteeing good coverage of the embedding space. Falls back to stratified
    sampling when no "embedding" feature is present.
    """

    def select(self, items: List[dict], prune_ratio: float, seed: int = 0) -> List[int]:
        if not items:
            return []
        if prune_ratio >= 1.0:
            return [item["index"] for item in items]

        has_embeddings = all("embedding" in item["features"] for item in items)
        if not has_embeddings:
            from .stratified import StratifiedStrategy
            return StratifiedStrategy().select(items, prune_ratio, seed)

        rng = np.random.default_rng(seed)
        n = len(items)
        k = max(1, round(n * prune_ratio))

        if k >= n:
            return [item["index"] for item in items]

        embeddings = np.array(
            [item["features"]["embedding"] for item in items], dtype=float
        )

        first = int(rng.integers(0, n))
        chosen = [first]

        min_dists = np.sum((embeddings - embeddings[first]) ** 2, axis=1)
        min_dists[first] = -1.0

        for _ in range(k - 1):
            next_pt = int(np.argmax(min_dists))
            chosen.append(next_pt)
            new_dists = np.sum((embeddings - embeddings[next_pt]) ** 2, axis=1)
            np.minimum(min_dists, new_dists, out=min_dists)
            min_dists[next_pt] = -1.0

        return [items[i]["index"] for i in chosen]
