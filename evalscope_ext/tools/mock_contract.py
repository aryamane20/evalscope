"""mock_contract: demonstrate the live_code_bench_pruned contract end-to-end
using a zero-cost mock model (evalscope's built-in MockLLM / mock_llm eval_type).

This script:
  1. Creates a 30-record synthetic LCB dataset in the Arrow cache location that
     evalscope's RemoteDataLoader finds on first access, so no network call is
     needed.
  2. Runs the FULL benchmark (live_code_bench) with a fixed limit.
  3. Runs the PRUNED benchmark twice with identical pruning params, once using
     the FLAT dataset-args contract (matches the spec) and once using the
     NESTED extra_params contract, and confirms the kept-sample count is
     identical and reduced in both.
  4. Calls compare_runs and prints its output.

Usage (from repo root):
    python -m evalscope_ext.tools.mock_contract

Reports land under /tmp/evalscope_mock/ so they do not pollute the repo.
"""
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_FULL = Path('/tmp/evalscope_mock/full')
RESULTS_PRUNED_FLAT = Path('/tmp/evalscope_mock/pruned_flat')
RESULTS_PRUNED_NESTED = Path('/tmp/evalscope_mock/pruned_nested')
SUBSET = 'release_v5'
LIMIT = 20
PRUNE_RATIO = 0.2
STRATEGY = 'stratified'

# Imports below are after sys.path setup on line 26; E402 is intentional.
from evalscope.utils.io_utils import gen_hash, safe_filename  # noqa: E402
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR  # noqa: E402
from evalscope.config import TaskConfig, EvalType  # noqa: E402

_DATASET_ID = 'evalscope/livecodebench_code_generation_lite_parquet'
_SPLIT = 'test'
_SUBSET = SUBSET

_hash = gen_hash(f'{_DATASET_ID}{_SPLIT}{_SUBSET}None{{}}')

_probe_cfg = TaskConfig(datasets=['live_code_bench'], eval_type=EvalType.MOCK_LLM)
_base = (_probe_cfg.dataset_dir or DEFAULT_EVALSCOPE_CACHE_DIR)
_cache_dir = Path(_base) / 'datasets' / f'{safe_filename(_DATASET_ID)}-{_hash}'

print(f'[mock_contract] Dataset cache path: {_cache_dir}')

if not _cache_dir.exists():
    import datasets as hf_datasets

    print('[mock_contract] Creating 30-record synthetic LCB dataset...')

    records = []
    for i in range(30):
        test_cases = json.dumps([
            {'input': f'{i} {i + 1}', 'output': str(2 * i + 1)},
        ])
        records.append({
            'question_content': (
                f'Problem {i + 1}: Write a Python function that reads two integers '
                f'from stdin and prints their sum.'
            ),
            'starter_code': '',
            'public_test_cases': test_cases,
            'private_test_cases': test_cases,
            'metadata': json.dumps({'func_name': None}),
            'contest_date': f'2024-{(i % 6) + 1:02d}-15T00:00:00',
        })

    ds = hf_datasets.Dataset.from_list(records)
    _cache_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(_cache_dir))
    print(f'[mock_contract] Saved {len(records)} synthetic records -> {_cache_dir}')
else:
    print(f'[mock_contract] Using existing cached dataset at {_cache_dir}')

import logging  # noqa: E402
logging.getLogger('evalscope').setLevel(logging.INFO)

from evalscope.run import run_task  # noqa: E402


def _eval(dataset_name: str, work_dir: Path, dataset_args: dict, label: str) -> Path:
    """Run an eval with the given raw dataset-args and return the report path."""
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = TaskConfig(
        model='mockllm',
        eval_type=EvalType.MOCK_LLM,
        datasets=[dataset_name],
        dataset_args={dataset_name: dataset_args},
        limit=LIMIT,
        work_dir=str(work_dir),
        no_timestamp=True,
    )

    print(f'\n{"=" * 60}')
    print(f'[mock_contract] Running: {label}')
    print(f'  dataset      : {dataset_name}')
    print(f'  dataset_args : {dataset_args}')
    print(f'  limit        : {LIMIT}')
    print(f'  work_dir     : {work_dir}')
    print('=' * 60)

    run_task(cfg)

    report_files = sorted(work_dir.glob('reports/**/*.json'))
    if not report_files:
        raise RuntimeError(f'No report JSON found under {work_dir}/reports/')
    return report_files[0]


def _report_num(report_path: Path) -> int:
    """Return the evaluated-sample count (kept count after pruning)."""
    return json.loads(report_path.read_text()).get('num', -1)


full_report_path = _eval(
    dataset_name='live_code_bench',
    work_dir=RESULTS_FULL,
    dataset_args={'subset_list': [SUBSET]},
    label=f'Full benchmark (live_code_bench, limit={LIMIT})',
)
full_n = _report_num(full_report_path)
print(f'\n[mock_contract] Full report -> {full_report_path}  num={full_n}')

flat_report_path = _eval(
    dataset_name='live_code_bench_pruned',
    work_dir=RESULTS_PRUNED_FLAT,
    dataset_args={
        'pruning_strategy': STRATEGY,
        'prune_ratio': PRUNE_RATIO,
        'subset_list': [SUBSET],
    },
    label=f'Pruned, FLAT contract (strategy={STRATEGY}, prune_ratio={PRUNE_RATIO})',
)
flat_n = _report_num(flat_report_path)
print(f'\n[mock_contract] Pruned (flat) report -> {flat_report_path}  num={flat_n}')

nested_report_path = _eval(
    dataset_name='live_code_bench_pruned',
    work_dir=RESULTS_PRUNED_NESTED,
    dataset_args={
        'extra_params': {
            'pruning_strategy': STRATEGY,
            'prune_ratio': PRUNE_RATIO,
        },
        'subset_list': [SUBSET],
    },
    label=f'Pruned, NESTED contract (strategy={STRATEGY}, prune_ratio={PRUNE_RATIO})',
)
nested_n = _report_num(nested_report_path)
print(f'\n[mock_contract] Pruned (nested) report -> {nested_report_path}  num={nested_n}')

print(f'\n{"-" * 60}')
print(f'  Kept-sample counts (limit={LIMIT}, prune_ratio={PRUNE_RATIO}):')
print(f'  Full (live_code_bench)          : {full_n}')
print(f'  Pruned FLAT contract            : {flat_n}')
print(f'  Pruned NESTED contract          : {nested_n}')
print('-' * 60)

assert flat_n == nested_n, (
    f'FLAT ({flat_n}) and NESTED ({nested_n}) kept counts differ; '
    f'the flat contract is not producing real pruning.'
)
assert isinstance(flat_n, int) and 0 < flat_n < full_n, (
    f'Pruned count {flat_n} is not reduced below full count {full_n}.'
)
print(f'  PASS: both contracts kept {flat_n} samples (reduced from {full_n}).')

print(f'\n[mock_contract] Running compare_runs (flat vs full)...\n')

import subprocess  # noqa: E402
subprocess.run(
    [sys.executable, '-m', 'evalscope_ext.tools.compare_runs',
     '--full', str(RESULTS_FULL),
     '--pruned', str(RESULTS_PRUNED_FLAT),
     '--tolerance', '0.15'],
    capture_output=False,
    cwd=str(REPO_ROOT),
)

print(f'\n[mock_contract] Contract demonstrated successfully.')
print("""
Reviewer commands (substitute a real model):

  # Full eval
  evalscope eval \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --api-url http://localhost:8000/v1/chat/completions \\
    --api-key EMPTY \\
    --datasets live_code_bench \\
    --dataset-args '{"subset_list":["release_v6"]}' \\
    --work-dir ./results_full --no-timestamp

  # Pruned eval (flat contract, matches the spec)
  evalscope eval \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --api-url http://localhost:8000/v1/chat/completions \\
    --api-key EMPTY \\
    --datasets live_code_bench_pruned \\
    --dataset-args '{"pruning_strategy":"stratified","prune_ratio":0.2,"subset_list":["release_v6"]}' \\
    --work-dir ./results_pruned --no-timestamp

  # Compare
  python -m evalscope_ext.tools.compare_runs \\
    --full ./results_full --pruned ./results_pruned --tolerance 0.05
""")
