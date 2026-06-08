"""prune_sweep: ratio x strategy sweep for pruning validation.

Produces four outputs for each benchmark (live_code_bench, aa_lcr):

  1. Ratio x strategy table: ranking_preserved and per-model score diffs.
  2. Smallest ranking-preserving ratio per strategy (the headline figure).
  3. Bootstrap (1000 seeds): ranking-preservation probability and 95% CI on
     each model's pruned score at the chosen ratio.
  4. Bias table: mean signed diff (pruned - full) per strategy/ratio.

Feature extraction (score-free in all cases):
  LCB     cross-model avg(total_tokens) from prediction files.
  AA-LCR  input_tokens from sample_metadata in review files.

Usage (run from repo root):
    python -m evalscope_ext.tools.prune_sweep \\
        --evals-dir "/path/to/Evals/Part 1" \\
        [--ratios 0.10 0.15 0.20 0.25 0.30] \\
        [--strategies stratified stratified_coreset difficulty_spread] \\
        [--n-seeds 1000] \\
        [--seed 42]
"""

import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _boot_engine():
    root = Path(__file__).resolve().parent.parent.parent
    pdir = root / 'evalscope' / 'benchmarks' / 'pruned' / 'pruning'
    pkg = '_pe_sweep'
    if f'{pkg}.base' not in sys.modules:
        stub = types.ModuleType(pkg)
        stub.__path__ = [str(pdir)]
        stub.__package__ = pkg
        sys.modules[pkg] = stub
        for name in ['base', 'stratified', 'coreset', 'difficulty', 'stratified_coreset']:
            spec = importlib.util.spec_from_file_location(f'{pkg}.{name}', pdir / f'{name}.py')
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg
            sys.modules[f'{pkg}.{name}'] = mod
            spec.loader.exec_module(mod)
    return sys.modules[f'{pkg}.base'].get_strategy


def _score(record: dict) -> Optional[float]:
    """Extract score; handles both evalscope output and shipped flat format."""
    ss = record.get('sample_score') or {}
    value = ss.get('value') or (ss.get('score') or {}).get('value')
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_model_scores(reviews_dir: Path, prefix: str) -> Dict[str, Dict[int, float]]:
    """Return {model: {problem_index: score}}."""
    result: Dict[str, Dict[int, float]] = {}
    for f in sorted(reviews_dir.glob(f'{prefix}__*.jsonl')):
        model = f.stem.split('__', 1)[1]
        scores: Dict[int, float] = {}
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            s = _score(rec)
            if s is not None:
                scores[rec.get('index', len(scores))] = s
        result[model] = scores
    return result


def _full_mean(model_scores: Dict[int, float]) -> float:
    v = list(model_scores.values())
    return sum(v) / len(v) if v else 0.0


def _pruned_mean(model_scores: Dict[int, float], kept: set) -> Tuple[float, int]:
    vals = [model_scores[i] for i in kept if i in model_scores]
    return (sum(vals) / len(vals), len(vals)) if vals else (0.0, 0)


def _quantile_items(index_val_pairs: List[Tuple[int, float]], n_buckets: int = 5) -> List[dict]:
    """Build pruning-engine items from (problem_index, difficulty_value) pairs."""
    vals = [v for _, v in index_val_pairs]
    sorted_v = sorted(vals)
    n = len(sorted_v)
    thresholds = [sorted_v[max(0, int(n * i / n_buckets))] for i in range(1, n_buckets)]
    min_v, max_v = sorted_v[0], sorted_v[-1]
    span = (max_v - min_v) or 1.0
    items = []
    for idx, v in index_val_pairs:
        q = sum(v >= t for t in thresholds)
        items.append({'index': idx, 'features': {
            'bucket': f'q{q}',
            'difficulty': (v - min_v) / span,
        }})
    return items


def _lcb_items(evals_dir: Path, n_buckets: int = 5) -> List[dict]:
    """LCB items: cross-model avg(total_tokens) from prediction files."""
    tokens: Dict[int, List[float]] = defaultdict(list)
    for f in sorted((evals_dir / 'predictions').glob('live_code_bench_v5__*.jsonl')):
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            usage = (rec.get('model_output') or {}).get('usage') or {}
            tok = usage.get('total_tokens')
            if tok and tok > 0:
                tokens[rec['index']].append(float(tok))
    pairs = [(idx, sum(vs) / len(vs)) for idx, vs in sorted(tokens.items())]
    return _quantile_items(pairs, n_buckets)


def _aalcr_items(evals_dir: Path, n_buckets: int = 5) -> List[dict]:
    """AA-LCR items: input_tokens from review sample_metadata (score-free)."""
    f = sorted((evals_dir / 'reviews').glob('aa_lcr__*.jsonl'))[0]
    pairs = []
    for line in f.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        idx = rec.get('index', len(pairs))
        meta = (rec.get('sample_score') or {}).get('sample_metadata') or {}
        tok = float(meta.get('input_tokens', 0) or 0)
        pairs.append((idx, tok))
    return _quantile_items(pairs, n_buckets)


def _run_sweep(
    items: List[dict],
    all_scores: Dict[str, Dict[int, float]],
    ratios: List[float],
    strategies: List[str],
    get_strategy,
    seed: int,
) -> Tuple[List[dict], Dict[str, float], List[str]]:
    """Return (rows, full_scores_by_model, full_rank)."""
    models = sorted(all_scores.keys())
    full_scores = {m: _full_mean(all_scores[m]) for m in models}
    full_rank = sorted(models, key=lambda m: -full_scores[m])

    rows = []
    for sname in strategies:
        for ratio in ratios:
            strat = get_strategy(sname)()
            kept = set(strat.select(items, ratio, seed=seed))
            n_kept = len(kept)

            pruned = {}
            for m in models:
                pm, _ = _pruned_mean(all_scores[m], kept)
                pruned[m] = pm

            pruned_rank = sorted(models, key=lambda m: -pruned[m])
            rank_ok = pruned_rank == full_rank
            diffs = {m: pruned[m] - full_scores[m] for m in models}
            mean_abs = sum(abs(d) for d in diffs.values()) / len(diffs)
            mean_signed = sum(diffs.values()) / len(diffs)

            rows.append({
                'strategy': sname,
                'ratio': ratio,
                'n_kept': n_kept,
                'rank_ok': rank_ok,
                'mean_abs': mean_abs,
                'mean_signed': mean_signed,
                'pruned': pruned,
                'diffs': diffs,
                'pruned_rank': pruned_rank,
            })

    return rows, full_scores, full_rank


def _bootstrap(
    items: List[dict],
    all_scores: Dict[str, Dict[int, float]],
    strategy_name: str,
    ratio: float,
    get_strategy,
    n_seeds: int = 1000,
) -> Tuple[float, Dict[str, Tuple[float, float, float]]]:
    """Sweep n_seeds random seeds; return (p_rank_ok, {model: (lo, hi, mean)})."""
    models = sorted(all_scores.keys())
    full_scores = {m: _full_mean(all_scores[m]) for m in models}
    full_rank = sorted(models, key=lambda m: -full_scores[m])

    per_model: Dict[str, List[float]] = defaultdict(list)
    rank_ok_count = 0

    for s in range(n_seeds):
        strat = get_strategy(strategy_name)()
        kept = set(strat.select(items, ratio, seed=s))
        pruned = {}
        for m in models:
            pm, _ = _pruned_mean(all_scores[m], kept)
            pruned[m] = pm
            per_model[m].append(pm)
        if sorted(models, key=lambda m: -pruned[m]) == full_rank:
            rank_ok_count += 1

    ci: Dict[str, Tuple[float, float, float]] = {}
    for m in models:
        sv = sorted(per_model[m])
        lo = sv[max(0, int(n_seeds * 0.025))]
        hi = sv[min(n_seeds - 1, int(n_seeds * 0.975))]
        mean = sum(per_model[m]) / n_seeds
        ci[m] = (lo, hi, mean)

    return rank_ok_count / n_seeds, ci


def _print_sweep(rows, full_scores, full_rank, models, W=88):
    header = (f"  {'Strategy':<22} {'Ratio':>6} {'n':>5}  {'Rank?':>5}"
              f"  {'Mean|d|':>8}  " + "  ".join(f"{m[:12]:>12}" for m in models))
    print(f"\n{'-'*W}")
    print(header)
    print(f"  {'':22} {'':6} {'':5}  {'':5}  {'full:':>8}  "
          + "  ".join(f"{full_scores[m]:>12.4f}" for m in models))
    print(f"{'-'*W}")
    prev_strat = None
    for r in rows:
        if prev_strat and r['strategy'] != prev_strat:
            print()
        prev_strat = r['strategy']
        rank_str = 'PASS' if r['rank_ok'] else 'FAIL'
        diff_cols = "  ".join(f"{r['diffs'][m]:>+12.4f}" for m in models)
        print(f"  {r['strategy']:<22} {r['ratio']:>6.2f} {r['n_kept']:>5}  "
              f"{rank_str:>5}  {r['mean_abs']:>8.4f}  {diff_cols}")
    print(f"{'-'*W}")


def _smallest_preserving(rows, strategies, ratios) -> Dict[str, Optional[float]]:
    result = {}
    for sname in strategies:
        found = None
        for r in rows:
            if r['strategy'] == sname and r['rank_ok']:
                if found is None or r['ratio'] < found:
                    found = r['ratio']
        result[sname] = found
    return result


def _print_bootstrap(strategy, ratio, n_kept, p, ci, full_scores, models, n_seeds):
    print(f"\n  Strategy: {strategy} at ratio={ratio:.2f}  (n~{n_kept})")
    print(f"  {'Model':<24} {'Full':>7}  {'Mean prnd':>9}  {'95% CI':>22}")
    print(f"  {'-'*70}")
    for m in models:
        lo, hi, mean = ci[m]
        print(f"  {m:<24} {full_scores[m]:>7.4f}  {mean:>9.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"\n  Ranking preserved in {p*100:.1f}% of {n_seeds} seed-sweep iterations")


def _print_bias(rows, strategies, ratios, models, W=80):
    print(f"\n  {'Strategy':<22} {'Metric':<14}" + "".join(f"  {r:.2f}" for r in ratios))
    print(f"  {'-'*W}")
    for sname in strategies:
        strategy_rows = {r['ratio']: r for r in rows if r['strategy'] == sname}
        cells = "".join(
            f"  {strategy_rows[rt]['mean_signed']:>+6.4f}" if rt in strategy_rows else "       N/A"
            for rt in ratios
        )
        print(f"  {sname:<22} {'mean signed d':<14}{cells}")
    print()
    last = ratios[-1]
    biases = {sname: abs(next(r['mean_signed'] for r in rows
                              if r['strategy'] == sname and r['ratio'] == last))
              for sname in strategies}
    best = min(biases, key=biases.get)
    print(f"  Least biased at ratio={last:.2f}: {best}  (|mean signed d|={biases[best]:.4f})")


def _run_benchmark(
    label: str,
    items: List[dict],
    all_scores: Dict[str, Dict[int, float]],
    ratios: List[float],
    strategies: List[str],
    get_strategy,
    seed: int,
    n_seeds: int,
    judge_note: str = '',
):
    models = sorted(all_scores.keys())
    full_scores = {m: _full_mean(all_scores[m]) for m in models}
    full_rank = sorted(models, key=lambda m: -full_scores[m])
    second_gap = sorted(full_scores.values())[-2] - sorted(full_scores.values())[-3] \
                 if len(full_scores) >= 3 else 0.0

    W = 90
    print(f"\n{'='*W}")
    print(f"  BENCHMARK: {label}   ({len(items)} problems, {len(models)} models)")
    fs_str = "  ".join(f"{m}={full_scores[m]:.4f}" for m in full_rank)
    print(f"  Full scores:  {fs_str}")
    print(f"  Full ranking: {' > '.join(m.split('-')[0] for m in full_rank)}")
    print(f"  Closest pair gap: {second_gap:.4f} ({sorted(models, key=lambda m: -full_scores[m])[-2].split('-')[0]} vs "
          f"{sorted(models, key=lambda m: -full_scores[m])[-1].split('-')[0]})")
    if judge_note:
        print(f"  Note: {judge_note}")
    print(f"{'='*W}")

    print(f"\n  1. RATIO x STRATEGY SWEEP  (canonical seed={seed})")
    rows, _, _ = _run_sweep(items, all_scores, ratios, strategies, get_strategy, seed)
    _print_sweep(rows, full_scores, full_rank, models)

    print(f"\n  2. SMALLEST RANKING-PRESERVING RATIO")
    smallest = _smallest_preserving(rows, strategies, ratios)
    for sname in strategies:
        sv = smallest[sname]
        if sv is None:
            print(f"    {sname:<25} not preserved at any tested ratio")
        else:
            n_at = next(r['n_kept'] for r in rows if r['strategy'] == sname and r['ratio'] == sv)
            pct = sv * 100
            print(f"    {sname:<25} ratio={sv:.2f}  n~{n_at}  ({pct:.0f}% kept)")

    print(f"\n  3. BOOTSTRAP ({n_seeds} seeds): ranking preservation probability")
    for sname in strategies:
        sv = smallest[sname]
        if sv is None:
            sv = ratios[-1]
            print(f"\n  [Note] {sname}: ranking not preserved at any tested ratio; "
                  f"bootstrapping at ratio={sv:.2f} anyway.")
        n_at = next(r['n_kept'] for r in rows if r['strategy'] == sname and r['ratio'] == sv)
        p, ci = _bootstrap(items, all_scores, sname, sv, get_strategy, n_seeds)
        _print_bootstrap(sname, sv, n_at, p, ci, full_scores, models, n_seeds)
        sorted_m = sorted(models, key=lambda m: -full_scores[m])
        if len(sorted_m) >= 2:
            m1, m2 = sorted_m[-2], sorted_m[-1]
            lo1, hi1, _ = ci[m1]
            lo2, hi2, _ = ci[m2]
            if hi2 > lo1:
                print(f"\n  [Warn] CIs for {m1.split('-')[0]} and {m2.split('-')[0]} overlap; "
                      f"not reliably separable at n~{n_at}")

    print(f"\n  4. BIAS CHECK (mean signed d: pruned - full per strategy/ratio)")
    _print_bias(rows, strategies, ratios, models)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m evalscope_ext.tools.prune_sweep',
        description='Ratio x strategy sweep for pruning validation.',
    )
    parser.add_argument('--evals-dir', type=Path, required=True)
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument('--strategies', nargs='+',
                        default=['stratified', 'stratified_coreset', 'difficulty_spread'])
    parser.add_argument('--n-seeds', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    get_strategy = _boot_engine()
    rev_dir = args.evals_dir / 'reviews'

    lcb_items = _lcb_items(args.evals_dir)
    lcb_scores = _load_model_scores(rev_dir, 'live_code_bench_v5')
    _run_benchmark(
        label='live_code_bench_v5',
        items=lcb_items,
        all_scores=lcb_scores,
        ratios=args.ratios,
        strategies=args.strategies,
        get_strategy=get_strategy,
        seed=args.seed,
        n_seeds=args.n_seeds,
        judge_note='',
    )

    aalcr_items = _aalcr_items(args.evals_dir)
    aalcr_scores = _load_model_scores(rev_dir, 'aa_lcr')
    _run_benchmark(
        label='aa_lcr',
        items=aalcr_items,
        all_scores=aalcr_scores,
        ratios=args.ratios,
        strategies=args.strategies,
        get_strategy=get_strategy,
        seed=args.seed,
        n_seeds=args.n_seeds,
        judge_note='scores from LLM judge; some variance is judge noise, not model variance',
    )


if __name__ == '__main__':
    main()
