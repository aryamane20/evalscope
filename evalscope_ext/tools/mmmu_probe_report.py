"""mmmu_probe_report: compute encoder_gap and degradation curve from an mmmu_pruned probe run.

Given the work_dir produced by an mmmu_pruned evaluation with probe_mode != 'none',
reads the review JSONL files and computes per-group metrics:

  encoder_gap
    = mean(acc[text_only]) - mean(acc[image])  per perception/knowledge group
    Large positive gap: model can reason but cannot see (encoder weakness).
    Near zero: image and text-only perform similarly.
    Large negative: model relies on images (expected for vision-heavy tasks).

  degradation_curve
    acc at each perturbation level (pert_scale75, pert_scale50, pert_scale25
    or pert_blur1, pert_blur2, pert_blur4) per group.
    Graceful (stable) = strong encoder; collapse = weak.

Review JSONL files are read from:
  <work_dir>/reviews/<model_name>/mmmu_pruned_<subset>.jsonl

Each line is a ReviewResult JSON object with:
  sample_score.score.value        dict e.g. {"acc": 1.0}
  sample_score.sample_metadata    dict containing 'variant' and 'img_group'

Usage:
    python -m evalscope_ext.tools.mmmu_probe_report --work-dir /tmp/probe_run/

    # or programmatically:
    from evalscope_ext.tools.mmmu_probe_report import run_report
    run_report('/tmp/probe_run/')
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PERT_ORDER_SCALE = ['pert_scale75', 'pert_scale50', 'pert_scale25']
PERT_ORDER_BLUR = ['pert_blur1', 'pert_blur2', 'pert_blur4']


def _find_review_files(work_dir: Path) -> List[Path]:
    """Return all review JSONL files for mmmu_pruned under <work_dir>/reviews/."""
    reviews_root = work_dir / 'reviews'
    if not reviews_root.exists():
        raise FileNotFoundError(
            f"Reviews directory not found: {reviews_root}\n"
            f"Did the eval run write to '{work_dir}'?\n"
            f"Expected: {work_dir}/reviews/<model_name>/mmmu_pruned_<subset>.jsonl"
        )
    pattern = str(reviews_root / '**' / 'mmmu_pruned_*.jsonl')
    found = sorted(glob.glob(pattern, recursive=True))
    if not found:
        raise FileNotFoundError(
            f"No mmmu_pruned review files found under '{reviews_root}'.\n"
            f"Run an eval with probe_mode != 'none' first."
        )
    return [Path(p) for p in found]


def _load_records(review_files: List[Path]) -> List[dict]:
    """Parse all review JSONL lines and return a flat list of dicts."""
    records = []
    for path in review_files:
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except OSError as exc:
            print(f'[WARN] Cannot read {path}: {exc}', file=sys.stderr)
            continue
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f'[WARN] {path}:{lineno} JSON error: {exc}', file=sys.stderr)
    return records


def _extract_row(record: dict) -> Optional[Tuple[str, str, float]]:
    """Extract (variant, img_group, acc) from a single ReviewResult dict.

    Returns None if required fields are missing.
    """
    try:
        ss = record['sample_score']
        meta = ss.get('sample_metadata') or {}
        variant = meta.get('variant')
        img_group = meta.get('img_group')
        score_value = ss['score']['value']
        acc = float(next(iter(score_value.values())))
    except (KeyError, TypeError, StopIteration, ValueError):
        return None

    if not variant or not img_group:
        return None

    return variant, img_group, acc


def compute_report(records: List[dict]) -> dict:
    """Aggregate review records into encoder_gap and degradation_curve dicts.

    Returns a dict with keys:
      'encoder_gap'        {img_group: float | None}
      'degradation_curve'  {img_group: {pert_level: float | None}}
      'sample_counts'      {img_group: {variant: int}}
    """
    acc_by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        row = _extract_row(rec)
        if row is None:
            continue
        variant, img_group, acc = row
        acc_by[img_group][variant].append(acc)

    groups = sorted(acc_by.keys())

    encoder_gap: Dict[str, Optional[float]] = {}
    sample_counts: Dict[str, Dict[str, int]] = {}

    for group in groups:
        variants = acc_by[group]
        image_accs = variants.get('image', [])
        text_accs = variants.get('text_only', [])
        if image_accs and text_accs:
            encoder_gap[group] = _mean(text_accs) - _mean(image_accs)
        else:
            encoder_gap[group] = None
        sample_counts[group] = {v: len(lst) for v, lst in variants.items()}

    pert_levels = _detect_pert_levels(acc_by)

    degradation_curve: Dict[str, Dict[str, Optional[float]]] = {}
    for group in groups:
        variants = acc_by[group]
        image_acc = _mean(variants['image']) if variants.get('image') else None
        curve: Dict[str, Optional[float]] = {'image': image_acc}
        for level in pert_levels:
            lst = variants.get(level, [])
            curve[level] = _mean(lst) if lst else None
        degradation_curve[group] = curve

    return {
        'encoder_gap': encoder_gap,
        'degradation_curve': degradation_curve,
        'sample_counts': sample_counts,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _detect_pert_levels(acc_by: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """Return perturbation levels found in the data, in the expected order."""
    all_variants: set = set()
    for group_data in acc_by.values():
        all_variants.update(group_data.keys())
    pert_variants = {v for v in all_variants if v.startswith('pert_')}
    if any('scale' in v for v in pert_variants):
        return [v for v in PERT_ORDER_SCALE if v in pert_variants]
    if any('blur' in v for v in pert_variants):
        return [v for v in PERT_ORDER_BLUR if v in pert_variants]
    return sorted(pert_variants)


def print_report(report: dict, work_dir: Optional[Path] = None) -> None:
    """Print a formatted probe report to stdout."""
    W = 70
    encoder_gap = report['encoder_gap']
    degradation_curve = report['degradation_curve']
    sample_counts = report['sample_counts']

    print()
    print('=' * W)
    print('  MMMU IMAGE ENCODER PROBE REPORT')
    if work_dir:
        print(f'  work_dir: {work_dir}')
    print('=' * W)

    has_gap = any(v is not None for v in encoder_gap.values())
    has_curve = any(
        any(v is not None for k, v in curve.items() if k != 'image')
        for curve in degradation_curve.values()
    )

    def _fmt(v):
        return f'{v:.4f}' if v is not None else '  n/a  '

    if has_gap:
        print()
        print('  TEXT-ONLY CONTROL -- encoder_gap = acc(text_only) - acc(image)')
        print('  Positive gap: model can reason but cannot see (encoder weakness).')
        print()
        print(f"  {'Group':<16} {'acc(image)':>12} {'acc(text_only)':>15} {'encoder_gap':>13}  n(image)  n(text)")
        print(f"  {'-' * 66}")
        for group in sorted(encoder_gap.keys()):
            variants = sample_counts.get(group, {})
            acc_img_list = _get_accs_from_report(report, group, 'image')
            acc_txt_list = _get_accs_from_report(report, group, 'text_only')
            a_img = _mean(acc_img_list) if acc_img_list else None
            a_txt = _mean(acc_txt_list) if acc_txt_list else None
            gap_val = (a_txt - a_img) if (a_img is not None and a_txt is not None) else None
            n_img = variants.get('image', 0)
            n_txt = variants.get('text_only', 0)
            print(
                f"  {group:<16} {_fmt(a_img):>12} {_fmt(a_txt):>15} "
                f"{_fmt(gap_val):>13}  {n_img:>8}  {n_txt:>6}"
            )

    if has_curve:
        print()
        print('  PERTURBATION DEGRADATION -- acc at each image quality level')
        print('  Graceful curve = strong encoder; steep drop = weak encoder.')
        print()

        for group in sorted(degradation_curve.keys()):
            curve = degradation_curve[group]
            levels = [k for k in curve if k != 'image']
            if not levels:
                continue
            print(f'  Group: {group}')
            print(f"  {'Level':<16} {'acc':>8}  {'n':>6}")
            print(f"  {'-' * 34}")

            for level_key in ['image'] + levels:
                acc_val = curve.get(level_key)
                n = sample_counts.get(group, {}).get(level_key, 0)
                label = level_key if level_key != 'image' else 'image (baseline)'
                print(f"  {label:<16} {_fmt(acc_val):>8}  {n:>6}")
            print()

    if not has_gap and not has_curve:
        print()
        print('  No probe variants found in the review files.')
        print('  Run with probe_mode="text_control", "perturbation", or "both".')

    print('=' * W)
    print()


def _get_accs_from_report(report: dict, group: str, variant: str) -> List[float]:
    """Recover per-sample acc list from the sample_counts structure.

    The compute_report function only stores aggregates; this helper is used
    only in print_report which recomputes from the raw aggregates.  For a
    clean separation the raw accs are re-extracted directly.
    """
    return report.get('_raw_accs', {}).get(group, {}).get(variant, [])


def _compute_report_with_raw(records: List[dict]) -> dict:
    """Like compute_report but also embeds _raw_accs for print_report."""
    acc_by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        row = _extract_row(rec)
        if row is None:
            continue
        variant, img_group, acc = row
        acc_by[img_group][variant].append(acc)

    groups = sorted(acc_by.keys())
    encoder_gap: Dict[str, Optional[float]] = {}
    sample_counts: Dict[str, Dict[str, int]] = {}

    for group in groups:
        variants = acc_by[group]
        image_accs = variants.get('image', [])
        text_accs = variants.get('text_only', [])
        if image_accs and text_accs:
            encoder_gap[group] = _mean(text_accs) - _mean(image_accs)
        else:
            encoder_gap[group] = None
        sample_counts[group] = {v: len(lst) for v, lst in variants.items()}

    pert_levels = _detect_pert_levels(acc_by)
    degradation_curve: Dict[str, Dict[str, Optional[float]]] = {}
    for group in groups:
        variants = acc_by[group]
        image_acc = _mean(variants['image']) if variants.get('image') else None
        curve: Dict[str, Optional[float]] = {'image': image_acc}
        for level in pert_levels:
            lst = variants.get(level, [])
            curve[level] = _mean(lst) if lst else None
        degradation_curve[group] = curve

    return {
        'encoder_gap': encoder_gap,
        'degradation_curve': degradation_curve,
        'sample_counts': sample_counts,
        '_raw_accs': {g: dict(d) for g, d in acc_by.items()},
    }


def run_report(work_dir) -> dict:
    """Load reviews from work_dir and print the probe report. Returns the report dict."""
    work_dir = Path(work_dir)
    files = _find_review_files(work_dir)
    print(f'[mmmu_probe_report] Found {len(files)} review file(s):')
    for f in files:
        print(f'  {f}')
    records = _load_records(files)
    print(f'[mmmu_probe_report] Loaded {len(records)} review records.')
    report = _compute_report_with_raw(records)
    print_report(report, work_dir=work_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m evalscope_ext.tools.mmmu_probe_report',
        description='Compute encoder_gap and degradation curve from an mmmu_pruned probe run.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--work-dir', type=Path, required=True,
        metavar='DIR',
        help='Work dir written by the evalscope eval probe run.',
    )
    args = parser.parse_args()
    try:
        run_report(args.work_dir)
    except FileNotFoundError as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f'[ERROR] Unexpected error: {exc}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
