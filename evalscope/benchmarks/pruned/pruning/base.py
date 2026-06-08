from abc import ABC, abstractmethod
from typing import Dict, List, Type


class PruningStrategy(ABC):
    """Abstract base for all pruning strategies.

    Each strategy receives a list of items (each with an "index" int and a
    "features" dict) and returns the indices of the items to keep.
    Implementations must be deterministic given the same seed and must never
    read any score / pass / acc field.
    """

    @abstractmethod
    def select(self, items: List[dict], prune_ratio: float, seed: int = 0) -> List[int]:
        """Return the indices (item["index"]) of items to keep.

        Args:
            items:       List of {"index": int, "features": dict}.
            prune_ratio: Fraction of items to KEEP (0 < prune_ratio <= 1).
            seed:        RNG seed for determinism.

        Returns:
            List of kept item["index"] values (not positional indices).
        """


REGISTRY: Dict[str, Type[PruningStrategy]] = {}


def register_strategy(name: str):
    """Class decorator that registers a PruningStrategy under name."""

    def decorator(cls: Type[PruningStrategy]) -> Type[PruningStrategy]:
        if name in REGISTRY:
            raise ValueError(f"Strategy '{name}' is already registered.")
        REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str) -> Type[PruningStrategy]:
    """Return the PruningStrategy class registered under name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Strategy '{name}' not found. Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]
