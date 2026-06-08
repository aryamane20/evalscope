"""offline_analysis: validate pruning strategy against shipped model reviews.

Uses pre-computed predictions/reviews from real eval runs (Mode B) to simulate
what scores would have been if only the pruned subset were evaluated. No live
model calls or network access required.

Feature extraction strategy (score-free in all cases):

  live_code_bench  cross-model avg(total_tokens) from PREDICTION files.
                   Problems that demanded more tokens across all three models
                   are empirically harder; this is a stronger signal than
                   problem order (index) and is independent of any pass/fail.
                   Prediction files: <evals-dir>/predictions/<benchmark>*.jsonl
                   Field: model_output.usage.total_tokens

  aa_lcr           input_tokens from review sample_metadata.
                   Direct context-length measure; longer = harder.

  other/fallback   prompt length from messages[0].content if present,
                   else position-based coarse bucket.

Usage:

    python -m evalscope_ext.tools.offline_analysis \\
        --evals-dir ./shipped_evals/ \\
        --benchmark live_code_bench \\
        --strategy stratified_coreset \\
        --prune-ratio 0.1 \\
        [--seed 42] \\
        [--bootstrap 1000]

Expected layout of --evals-dir (two supported layouts):

  Layout A flat (shipped data):
    <evals-dir>/
      reviews/
        <benchmark>_<variant>__<model>.jsonl    one file per model
      predictions/
        <benchmark>_<variant>__<model>.jsonl

  Layout B evalscope per-model output dirs:
    <evals-dir>/
      <model_a>/
        reviews/<model_a>/<benchmark>_<subset>.jsonl
        predictions/<model_a>/<benchmark>_<subset>.jsonl
      <model_b>/
        ...

The tool NEVER reads score / pass / acc fields to derive pruning features.
"""

import argparse
import glob
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _load_pruning_engine() -> object:
    """Return the base module (with get_strategy) for the pruning engine."""
    _root = Path(__file__).resolve().parent.parent.parent
    _pruning_dir = _root / 'evalscope' / 'benchmarks' / 'pruned' / 'pruning'
    _pkg = '_pruning_engine_offline'

    if f'{_pkg}.base' not in sys.modules:
        stub = types.ModuleType(_pkg)
        stub.__path__ = [str(_pruning_dir)]
        stub.__package__ = _pkg
        sys.modules[_pkg] = stub

        for name in ['base', 'stratified', 'coreset', 'difficulty', 'stratified_coreset']:
            spec = importlib.util.spec_from_file_location(
                f'{_pkg}.{name}', _pruning_dir / f'{name}.py'
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = _pkg
            sys.modules[f'{_pkg}.{name}'] = mod
            spec.loader.exec_module(mod)

    return sys.modules[f'{_pkg}.base']


def _model_name_from_path(path: Path, benchmark: str) -> str:
    """Extract model name from a flat filename like <bench>__<model>.jsonl."""
    stem = path.stem
    if '__' in stem:
        return stem.split('__', 1)[1]
    return path.parent.name


def _find_model_dirs(evals_dir: Path, benchmark: str) -> Dict[str, List[Path]]:
    """Map model_name to list of review JSONL paths for benchmark.

    Layout A (flat): <evals_dir>/reviews/<benchmark>*.jsonl
    Layout B (per-model): <evals_dir>/<model>/reviews/**/<benchmark>_*.jsonl
    """
    result: Dict[str, List[Path]] = {}

    reviews_root = evals_dir / 'reviews'
    if reviews_root.is_dir():
        for f in sorted(reviews_root.glob(f'{benchmark}*.jsonl')):
            model = _model_name_from_path(f, benchmark)
            result.setdefault(model, []).append(f)
        if result:
            return result

    for subdir in sorted(evals_dir.iterdir()):
        if not subdir.is_dir():
            continue
        files = sorted(subdir.glob(f'reviews/**/{benchmark}_*.jsonl'))
        if files:
            result[subdir.name] = files

    if not result:
        files = sorted(evals_dir.glob(f'reviews/**/{benchmark}_*.jsonl'))
        if files:
            result[evals_dir.name] = files

    return result


def _find_prediction_files(evals_dir: Path, benchmark: str) -> List[Path]:
    """Find all prediction JSONL files for benchmark across all models.

    Returns a flat list (one entry per model-file); the caller aggregates.
    """
    found: List[Path] = []

    pred_root = evals_dir / 'predictions'
    if pred_root.is_dir():
        found.extend(sorted(pred_root.glob(f'{benchmark}*.jsonl')))
        if found:
            return found

    for subdir in sorted(evals_dir.iterdir()):
        if not subdir.is_dir():
            continue
        found.extend(sorted(subdir.glob(f'predictions/**/{benchmark}*.jsonl')))

    return found


def _load_records(files: List[Path]) -> List[dict]:
    """Load all JSONL records from a list of files."""
    records = []
    for path in files:
        try:
            for line in path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[WARN] Could not read {path}: {exc}', file=sys.stderr)
    return records


def _load_token_features(pred_files: List[Path]) -> Dict[int, float]:
    """Load cross-model avg(total_tokens) per problem index.

    Reads model_output.usage.total_tokens from every prediction file and
    averages across all models for each index. NEVER reads scores.

    Args:
        pred_files: list of prediction JSONL paths (all models).

    Returns:
        {index: avg_total_tokens_across_models}
    """
    tokens_by_index: Dict[int, List[float]] = defaultdict(list)

    for path in pred_files:
        try:
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                usage = (rec.get('model_output') or {}).get('usage') or {}
                tok = usage.get('total_tokens')
                if tok and tok > 0:
                    tokens_by_index[rec['index']].append(float(tok))
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[WARN] Could not read prediction file {path}: {exc}', file=sys.stderr)

    return {idx: sum(vs) / len(vs) for idx, vs in tokens_by_index.items()}


_TOK_BUCKET_LABELS = ['tok_very_short', 'tok_short', 'tok_medium', 'tok_long', 'tok_very_long']


def _build_token_items(
    records: List[dict],
    token_features: Dict[int, float],
    n_buckets: int = 5,
) -> Tuple[List[dict], List[Tuple[int, float, str]], List[float]]:
    """Build pruning-engine items using cross-model avg token counts.

    Returns:
        items          list of {index, features:{bucket,difficulty}} for get_strategy
        sample_rows    [(problem_index, avg_total_tokens, bucket)] for display
        thresholds     quantile cut-points between buckets (for display)
    """
    raw: List[Tuple[int, float]] = []
    for pos, rec in enumerate(records):
        idx = rec.get('index', pos)
        avg_tok = token_features.get(idx, 0.0)
        raw.append((idx, avg_tok))

    vals = sorted(avg_tok for _, avg_tok in raw)
    n = len(vals)
    labels = _TOK_BUCKET_LABELS[:n_buckets]

    thresholds = [vals[max(0, int(n * i / n_buckets))] for i in range(1, n_buckets)]

    min_tok, max_tok = vals[0], vals[-1]
    span = (max_tok - min_tok) or 1.0

    items = []
    sample_rows = []
    for i, (idx, avg_tok) in enumerate(raw):
        qidx = sum(avg_tok >= t for t in thresholds)
        bucket = labels[qidx]
        difficulty = (avg_tok - min_tok) / span
        items.append({'index': idx, 'features': {'bucket': bucket, 'difficulty': difficulty}})
        sample_rows.append((idx, avg_tok, bucket))

    return items, sample_rows, thresholds


def _record_to_item(record: dict, position: int) -> dict:
    """Convert a review record to a pruning-engine item (non-LCB path).

    Feature priority (score/pass/acc NEVER read):
      1. input_tokens from sample_metadata  (AA-LCR)
      2. Prompt length from messages[0].content
      3. Position-based coarse bucket (final fallback)
    """
    idx = record.get('index', position)

    ss = record.get('sample_score') or {}
    metadata = ss.get('sample_metadata') or {}
    if isinstance(metadata, dict):
        input_tokens = float(metadata.get('input_tokens', 0) or 0)
        if input_tokens > 0:
            thresholds = [80_000, 92_000, 100_000, 110_000]
            qidx = sum(input_tokens >= t for t in thresholds)
            labels = ['ctx_very_short', 'ctx_short', 'ctx_medium', 'ctx_long', 'ctx_very_long']
            return {
                'index': idx,
                'features': {
                    'bucket': f'aalcr__{labels[qidx]}',
                    'difficulty': min(1.0, input_tokens / 150_000),
                },
            }

    messages = record.get('messages', [])
    prompt_len = 0
    if messages:
        content = messages[0].get('content', '')
        if isinstance(content, str):
            prompt_len = len(content)
        elif isinstance(content, list):
            prompt_len = sum(
                len(part.get('text', '')) for part in content
                if isinstance(part, dict) and part.get('type') == 'text'
            )

    if prompt_len > 0:
        thresholds = [1_000, 3_000, 6_000, 10_000]
        qidx = sum(prompt_len >= t for t in thresholds)
        return {
            'index': idx,
            'features': {
                'bucket': f'len__q{qidx}',
                'difficulty': min(1.0, float(prompt_len) / 15_000),
            },
        }

    return {
        'index': idx,
        'features': {
            'bucket': f'pos_{position // 50}',
            'difficulty': min(1.0, float(position) / 300),
        },
    }


def _get_score(record: dict) -> Optional[float]:
    """Extract the primary numeric score from a review record.

    Handles two formats:
      evalscope output: sample_score.value  (dict or float)
      shipped flat:     sample_score.score.value  (nested Score object)
    """
    ss = record.get('sample_score') or {}
    value = ss.get('value')
    if value is None:
        score_obj = ss.get('score') or {}
        if isinstance(score_obj, dict):
            value = score_obj.get('value')
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    elif isinstance(value, (int, float)):
        return float(value)
    return None


def _mean_score(records: List[dict], kept_ids: Optional[Set[int]] = None) -> Tuple[float, int]:
    """Return (mean_score, n) over records, optionally restricted to kept_ids."""
    scores = []
    for rec in records:
        if kept_ids is not None and rec.get('index', -1) not in kept_ids:
            continue
        s = _get_score(rec)
        if s is not None:
            scores.append(s)
    return (sum(scores) / len(scores), len(scores)) if scores else (0.0, 0)


def _bootstrap_ci(
    scores: List[float],
    n_iter: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return (lo, hi) percentile bootstrap CI for the mean."""
    import random
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(
        sum(rng.choices(scores, k=n)) / n
        for _ in range(n_iter)
    )
    alpha = 1.0 - confidence
    lo = means[max(0, int(n_iter * alpha / 2))]
    hi = means[min(n_iter - 1, int(n_iter * (1 - alpha / 2)))]
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m evalscope_ext.tools.offline_analysis',
        description='Validate a pruning strategy against pre-computed review files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--evals-dir', type=Path, required=True, metavar='DIR',
                        help='Directory containing model eval output dirs')
    parser.add_argument('--benchmark', type=str, required=True, metavar='NAME',
                        help='Benchmark name (e.g. live_code_bench)')
    parser.add_argument('--strategy', type=str, default='stratified_coreset',
                        help='Pruning strategy name (default: stratified_coreset)')
    parser.add_argument('--prune-ratio', type=float, default=0.1,
                        help='Fraction of samples to keep (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed (default: 42)')
    parser.add_argument('--bootstrap', type=int, default=1000,
                        help='Bootstrap iterations for 95%% CI (0 to skip, default: 1000)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Max |full-pruned| to call PASS (default: 0.05)')

    args = parser.parse_args()

    base_mod = _load_pruning_engine()
    get_strategy = base_mod.get_strategy

    model_files = _find_model_dirs(args.evals_dir, args.benchmark)
    if not model_files:
        print(
            f"[ERROR] No review files for benchmark '{args.benchmark}' "
            f"found under '{args.evals_dir}'.",
            file=sys.stderr,
        )
        sys.exit(2)

    token_features: Dict[int, float] = {}
    is_lcb = 'live_code_bench' in args.benchmark.lower()
    if is_lcb:
        pred_files = _find_prediction_files(args.evals_dir, args.benchmark)
        if pred_files:
            token_features = _load_token_features(pred_files)
            print(
                f"  [INFO] Loaded token features from {len(pred_files)} prediction file(s): "
                f"{len(token_features)} indices, "
                f"avg_total_tokens range [{min(token_features.values()):.0f}, "
                f"{max(token_features.values()):.0f}]"
            )
        else:
            print(
                '  [WARN] No prediction files found for LCB; '
                'falling back to position-based features.',
                file=sys.stderr,
            )

    W = 72
    print()
    print('=' * W)
    print(f"  Offline Analysis: {args.benchmark}")
    feature_desc = ('cross-model avg total_tokens' if token_features
                    else 'position / metadata fallback')
    print(f"  Feature source: {feature_desc}")
    print(f"  Strategy: {args.strategy}  |  Prune ratio: {args.prune_ratio}  |  Seed: {args.seed}")
    print('=' * W)

    shared_items: Optional[List[dict]] = None
    sample_rows_display: Optional[List[Tuple[int, float, str]]] = None
    thresholds_display: Optional[List[float]] = None

    if token_features:
        first_model = sorted(model_files.keys())[0]
        first_records = _load_records(model_files[first_model])
        shared_items, sample_rows_display, thresholds_display = _build_token_items(
            first_records, token_features, n_buckets=5
        )

        print(f"\n  Token-count quantile boundaries (avg across {len(_find_prediction_files(args.evals_dir, args.benchmark))} models):")
        labels = _TOK_BUCKET_LABELS
        for i, t in enumerate(thresholds_display):
            print(f"    {labels[i]} / {labels[i+1]} boundary: {t:>8,.0f} tokens")

        from collections import Counter
        all_b = Counter(it['features']['bucket'] for it in shared_items)
        print(f"\n  Bucket distribution ({len(shared_items)} problems):")
        print(f"  {'Bucket':<22} {'Count':>6}  {'%':>6}")
        print(f"  {'-'*38}")
        for b in labels:
            c = all_b.get(b, 0)
            print(f"  {b:<22} {c:>6}  {c/len(shared_items)*100:>5.1f}%")

        print(f"\n  Sample rows (index, avg_total_tokens, bucket):")
        print(f"  {'index':>6}  {'avg_total_tok':>14}  bucket")
        print(f"  {'-'*44}")
        shown: Dict[str, int] = defaultdict(int)
        for idx, avg_tok, bucket in sample_rows_display:
            if shown[bucket] < 3:
                print(f"  {idx:>6}  {avg_tok:>14,.0f}  {bucket}")
                shown[bucket] += 1

    results: Dict[str, dict] = {}

    for model_name, files in sorted(model_files.items()):
        records = _load_records(files)
        if not records:
            print(f"\n  [WARN] No records for model '{model_name}'; skipping.")
            continue

        if shared_items is not None:
            items = shared_items
        else:
            items = [_record_to_item(r, i) for i, r in enumerate(records)]

        strategy = get_strategy(args.strategy)()
        kept_set: Set[int] = set(strategy.select(items, args.prune_ratio, seed=args.seed))

        kept_problem_ids = kept_set

        def _score_with_ids(recs, ids):
            scores = []
            for r in recs:
                if ids is not None and r.get('index', -1) not in ids:
                    continue
                s = _get_score(r)
                if s is not None:
                    scores.append(s)
            return (sum(scores) / len(scores), len(scores)) if scores else (0.0, 0)

        full_score, full_n = _score_with_ids(records, None)
        pruned_score, pruned_n = _score_with_ids(records, kept_problem_ids)
        delta = abs(full_score - pruned_score)
        ok = delta <= args.tolerance

        ci_str = ''
        if args.bootstrap > 0 and pruned_n > 0:
            pruned_scores_list = [
                _get_score(r) for r in records
                if r.get('index', -1) in kept_problem_ids and _get_score(r) is not None
            ]
            lo, hi = _bootstrap_ci(
                pruned_scores_list, n_iter=args.bootstrap, seed=args.seed
            )
            ci_str = f'  95% CI [{lo:.4f}, {hi:.4f}]'

        print(f"\n  Model : {model_name}")
        print(f"    Full  : {full_score:.4f}  (n={full_n})")
        print(f"    Pruned: {pruned_score:.4f}  (n={pruned_n}){ci_str}")
        print(f"    |delta|: {delta:.4f}  {'PASS' if ok else 'FAIL'}")

        results[model_name] = {
            'full_score': full_score,
            'pruned_score': pruned_score,
            'full_n': full_n,
            'pruned_n': pruned_n,
        }

    if len(results) >= 2:
        full_rank = sorted(results, key=lambda m: -results[m]['full_score'])
        pruned_rank = sorted(results, key=lambda m: -results[m]['pruned_score'])
        rank_ok = full_rank == pruned_rank

        print(f"\n  {'-'*68}")
        print(f"  Ranking (all models)")
        print(f"    Full  : {full_rank}")
        print(f"    Pruned: {pruned_rank}")
        print(f"    Preserved: {'PASS' if rank_ok else 'FAIL'}")

        print(f"\n  Leave-one-model-out ranking check:")
        loo_all_pass = True
        for left_out in sorted(results):
            remaining = {m: v for m, v in results.items() if m != left_out}
            if len(remaining) < 2:
                continue
            fr = sorted(remaining, key=lambda m: -remaining[m]['full_score'])
            pr = sorted(remaining, key=lambda m: -remaining[m]['pruned_score'])
            loo_ok = fr == pr
            loo_all_pass = loo_all_pass and loo_ok
            print(f"    leave out '{left_out}': {'PASS' if loo_ok else 'FAIL'}")
        print(f"    Overall: {'PASS' if loo_all_pass else 'FAIL'}")

    print(f"\n{'='*W}\n")


if __name__ == '__main__':
    main()
