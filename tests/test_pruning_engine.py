"""Tests for the standalone pruning engine.

The pruning modules are loaded directly via importlib under a private package
name (_pruning_engine) to avoid triggering evalscope/__init__.py, which
requires optional deps (modelscope) not needed by the engine itself.

All strategies are tested without evalscope wiring, no network, no model calls.
"""
import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PRUNING_DIR = (
    Path(__file__).parent.parent / "evalscope" / "benchmarks" / "pruned" / "pruning"
)
_PKG = "_pruning_engine"

if _PKG not in sys.modules:
    _stub = types.ModuleType(_PKG)
    _stub.__path__ = [str(_PRUNING_DIR)]
    _stub.__package__ = _PKG
    sys.modules[_PKG] = _stub

    for _name in ["base", "stratified", "coreset", "difficulty", "stratified_coreset"]:
        _spec = importlib.util.spec_from_file_location(
            f"{_PKG}.{_name}",
            _PRUNING_DIR / f"{_name}.py",
        )
        _mod = importlib.util.module_from_spec(_spec)
        _mod.__package__ = _PKG
        sys.modules[f"{_PKG}.{_name}"] = _mod
        _spec.loader.exec_module(_mod)

_base_mod = sys.modules[f"{_PKG}.base"]
get_strategy = _base_mod.get_strategy
PruningStrategy = _base_mod.PruningStrategy
REGISTRY = _base_mod.REGISTRY
register_strategy = _base_mod.register_strategy

N_ITEMS = 300
N_BUCKETS = 5
EMBEDDING_DIM = 10
DATA_SEED = 42


def make_items(
    n: int = N_ITEMS,
    n_buckets: int = N_BUCKETS,
    dim: int = EMBEDDING_DIM,
    seed: int = DATA_SEED,
    include_embeddings: bool = True,
) -> list:
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    bucket_names = [f"bucket_{i}" for i in range(n_buckets)]
    items = []
    for i in range(n):
        features = {
            "bucket": bucket_names[i % n_buckets],
            "difficulty": rng_py.random(),
        }
        if include_embeddings:
            features["embedding"] = rng_np.random(dim).tolist()
        items.append({"index": i, "features": features})
    return items


ALL_STRATEGIES = ["stratified", "coreset", "difficulty_spread", "stratified_coreset"]


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_prune_ratio_approx_10_percent(strategy_name):
    """prune_ratio=0.1 on 300 items should return ~30 items (±5 tolerance)."""
    items = make_items()
    kept = get_strategy(strategy_name)().select(items, prune_ratio=0.1, seed=0)
    expected = round(N_ITEMS * 0.1)
    assert abs(len(kept) - expected) <= 5, (
        f"{strategy_name}: expected ~{expected}, got {len(kept)}"
    )


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_deterministic_same_seed(strategy_name):
    """Two calls with identical seed must return identical index lists."""
    items = make_items()
    s = get_strategy(strategy_name)()
    assert sorted(s.select(items, prune_ratio=0.1, seed=42)) == \
           sorted(s.select(items, prune_ratio=0.1, seed=42)), \
        f"{strategy_name}: results differ across runs with the same seed"


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_different_seeds_produce_different_results(strategy_name):
    """Different seeds should (almost always) produce different subsets."""
    items = make_items()
    s = get_strategy(strategy_name)()
    kept0 = sorted(s.select(items, prune_ratio=0.1, seed=0))
    kept1 = sorted(s.select(items, prune_ratio=0.1, seed=999))
    assert kept0 != kept1, (
        f"{strategy_name}: seeds 0 and 999 produced identical subsets"
    )


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_all_returned_indices_are_valid(strategy_name):
    """Returned indices must be valid item['index'] values with no duplicates."""
    items = make_items()
    valid = {item["index"] for item in items}
    kept = get_strategy(strategy_name)().select(items, prune_ratio=0.1, seed=0)
    assert len(kept) == len(set(kept)), f"{strategy_name}: duplicate indices"
    assert all(idx in valid for idx in kept), f"{strategy_name}: out-of-range index"


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_prune_ratio_1_keeps_all(strategy_name):
    """prune_ratio=1.0 must return all items."""
    items = make_items(n=50)
    kept = get_strategy(strategy_name)().select(items, prune_ratio=1.0, seed=0)
    assert sorted(kept) == sorted(item["index"] for item in items)


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_empty_input_returns_empty(strategy_name):
    assert get_strategy(strategy_name)().select([], prune_ratio=0.1, seed=0) == []


def _proportion_check(items, strategy_name, tolerance=0.10):
    kept_set = set(get_strategy(strategy_name)().select(items, prune_ratio=0.1, seed=0))
    full: dict = {}
    kept: dict = {}
    for item in items:
        b = item["features"]["bucket"]
        full[b] = full.get(b, 0) + 1
        if item["index"] in kept_set:
            kept[b] = kept.get(b, 0) + 1
    n, n_kept = len(items), len(kept_set)
    for b, cnt in full.items():
        delta = abs(cnt / n - kept.get(b, 0) / n_kept)
        assert delta <= tolerance, (
            f"bucket '{b}': full={cnt/n:.2f}, kept={kept.get(b, 0)/n_kept:.2f}, "
            f"delta={delta:.2f} > {tolerance}"
        )


def test_stratified_preserves_category_proportions():
    """Stratified must mirror full-set bucket distribution within ±10 pp."""
    _proportion_check(make_items(), "stratified")


def test_stratified_coreset_preserves_category_proportions():
    """stratified_coreset inherits stratification, so distribution holds too."""
    _proportion_check(make_items(), "stratified_coreset")


def test_coreset_fallback_without_embeddings():
    items = make_items(include_embeddings=False)
    kept = get_strategy("coreset")().select(items, prune_ratio=0.1, seed=0)
    assert abs(len(kept) - 30) <= 5


def test_stratified_coreset_fallback_without_embeddings():
    items = make_items(include_embeddings=False)
    kept = get_strategy("stratified_coreset")().select(items, prune_ratio=0.1, seed=0)
    assert abs(len(kept) - 30) <= 5


def test_registry_contains_all_strategies():
    for name in ALL_STRATEGIES:
        assert name in REGISTRY, f"'{name}' missing from REGISTRY"


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        get_strategy("nonexistent_strategy")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        @register_strategy("stratified")
        class _Dup(PruningStrategy):
            def select(self, items, prune_ratio, seed=0):
                return []
