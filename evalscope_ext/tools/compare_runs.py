"""compare_runs: official reviewer contract for pruning validation.

Primary usage (the exact contract a reviewer runs):

    evalscope eval --model <m> --datasets live_code_bench     --output ./results_full/
    evalscope eval --model <m> --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy":"stratified_coreset","prune_ratio":0.1}' \\
        --output ./results_pruned/
    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ --pruned ./results_pruned/

Output directory layout expected (written by evalscope eval):

    <output_dir>/
      reports/
        <model_name>/
          <benchmark_name>.json   ← Report JSON (score, num, metrics[...])

Report JSON key fields:
    score          float  headline score (first metric, micro-mean over subsets)
    num            int    total evaluated samples
    model_name     str
    dataset_name   str

Exit codes: 0 = PASS (abs(full - pruned) <= tolerance), 1 = FAIL, 2 = error.
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def _find_reports(output_dir: Path) -> Dict[str, Tuple[dict, Path]]:
    """Return all report JSON files found under <output_dir>/reports/.

    Keys are the report's dataset_name field. Raises FileNotFoundError if no
    reports are present.
    """
    reports_root = output_dir / 'reports'
    if not reports_root.exists():
        raise FileNotFoundError(
            f"Directory '{reports_root}' does not exist.\n"
            f"Did the evalscope eval run write to '{output_dir}'?\n"
            f"Expected structure: {output_dir}/reports/<model>/<benchmark>.json"
        )

    found: Dict[str, Tuple[dict, Path]] = {}
    for json_path in sorted(reports_root.rglob('*.json')):
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            key = data.get('dataset_name', json_path.stem)
            found[key] = (data, json_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not read {json_path}: {exc}", file=sys.stderr)

    if not found:
        raise FileNotFoundError(
            f"No readable report JSON files under '{reports_root}'.\n"
            f"Expected: {output_dir}/reports/<model_name>/<benchmark_name>.json"
        )
    return found


def _pick_report(output_dir: Path, label: str) -> Tuple[dict, Path]:
    """Return (report_dict, report_path) for the given output dir.

    If multiple reports exist (multi-benchmark run), prints a warning and
    returns the first one alphabetically by dataset_name.
    """
    reports = _find_reports(output_dir)
    if len(reports) > 1:
        names = sorted(reports.keys())
        print(
            f"[WARN] {label}: multiple reports found {names}; "
            f"using '{names[0]}'. Pass --benchmark to select one.",
            file=sys.stderr,
        )
    key = sorted(reports.keys())[0]
    return reports[key]


def _pick_report_by_name(output_dir: Path, benchmark: str, label: str) -> Tuple[dict, Path]:
    """Return the report matching benchmark or raise FileNotFoundError."""
    reports = _find_reports(output_dir)
    if benchmark in reports:
        return reports[benchmark]
    matches = [k for k in reports if k.startswith(benchmark) or benchmark in k]
    if len(matches) == 1:
        return reports[matches[0]]
    raise FileNotFoundError(
        f"{label}: benchmark '{benchmark}' not found in {output_dir}/reports/.\n"
        f"Available: {sorted(reports.keys())}"
    )


def compare(
    full_dir: Path,
    pruned_dir: Path,
    tolerance: float = 0.05,
    benchmark: str | None = None,
) -> int:
    """Load reports, print table, return exit code (0=PASS, 1=FAIL)."""

    if benchmark:
        full_report, full_path = _pick_report_by_name(full_dir, benchmark, '--full')
        pruned_report, pruned_path = _pick_report_by_name(pruned_dir, benchmark + '_pruned', '--pruned')
        try:
            pruned_report, pruned_path = _pick_report_by_name(pruned_dir, benchmark + '_pruned', '--pruned')
        except FileNotFoundError:
            pruned_report, pruned_path = _pick_report_by_name(pruned_dir, benchmark, '--pruned')
    else:
        full_report, full_path = _pick_report(full_dir, '--full')
        pruned_report, pruned_path = _pick_report(pruned_dir, '--pruned')

    full_score: float = full_report.get('score', 0.0)
    pruned_score: float = pruned_report.get('score', 0.0)
    full_n: int = full_report.get('num', 0)
    pruned_n: int = pruned_report.get('num', 0)
    full_model: str = full_report.get('model_name', '?')
    full_bench: str = full_report.get('dataset_name', full_path.stem)
    pruned_bench: str = pruned_report.get('dataset_name', pruned_path.stem)
    abs_diff = abs(full_score - pruned_score)
    passed = abs_diff <= tolerance

    W = 68
    print()
    print('=' * W)
    print(f"  compare_runs: {full_bench}  vs  {pruned_bench}")
    print(f"  Model : {full_model}")
    print('-' * W)
    print(f"  {'Source':<18} {'Score':>9} {'Samples':>10}   Path")
    print(f"  {'-'*62}")
    print(f"  {'Full':<18} {full_score:>9.4f} {full_n:>10,}   {full_path}")
    print(f"  {'Pruned':<18} {pruned_score:>9.4f} {pruned_n:>10,}   {pruned_path}")
    print(f"  {'-'*62}")
    print(f"  {'|Δ| score':<18} {abs_diff:>9.4f}")
    print(f"  {'Tolerance':<18} {tolerance:>9.4f}")
    print('=' * W)

    verdict = 'PASS' if passed else 'FAIL'
    op = '<=' if passed else '>'
    print(
        f"\n  Decision preserved: {verdict}"
        f"  (|{full_score:.4f} - {pruned_score:.4f}| = {abs_diff:.4f} {op} {tolerance})"
    )
    print()

    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m evalscope_ext.tools.compare_runs',
        description='Compare full and pruned evalscope eval output directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--full', type=Path, required=True,
                        metavar='DIR', help='Output dir of the full benchmark eval run')
    parser.add_argument('--pruned', type=Path, required=True,
                        metavar='DIR', help='Output dir of the pruned benchmark eval run')
    parser.add_argument('--benchmark', type=str, default=None,
                        metavar='NAME',
                        help='Benchmark base name to select when multiple reports exist '
                             '(e.g. "live_code_bench" → looks for live_code_bench and '
                             'live_code_bench_pruned in the two dirs)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        metavar='T',
                        help='Max |full_score - pruned_score| to report PASS (default: 0.05)')

    args = parser.parse_args()

    try:
        code = compare(args.full, args.pruned, tolerance=args.tolerance, benchmark=args.benchmark)
        sys.exit(code)
    except FileNotFoundError as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f'[ERROR] Unexpected error: {exc}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
