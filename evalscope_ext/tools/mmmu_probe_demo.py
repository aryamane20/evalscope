"""mmmu_probe_demo: end-to-end demo of the MMMU image encoder probe with MockLLM.

This script runs in two phases:

  Phase 1 -- plumbing verification (standard MockLLM, fixed wrong answer):
    Confirms all 30 variants are produced, scored, and the reporter parses them.
    All accuracies are 0 (expected with a random-answer mock).

  Phase 2 -- DEMO SIGNAL (encoder-weakness simulation):
    Monkey-patches MockLLM.generate with a content-aware signal function that
    simulates two distinct encoder failure modes simultaneously:

      perception group (blue images):
        text_only -> correct  |  image/all-pert -> wrong
        => encoder_gap = acc(text_only) - acc(image) = 1.00 - 0.00 = +1.00
           (model can reason from text but cannot see)

      knowledge group (orange images):
        text_only -> correct  |  image/pert75 -> correct  |  pert50/pert25 -> wrong
        => encoder_gap = 1.00 - 1.00 = 0.00
           degradation curve: 1.00 -> 1.00 -> 0.00 -> 0.00  (collapses at 50%)

    The signal function detects variant type from message content:
      * No ContentImage + "[IMAGE:" in text  ->  text_only variant  ->  'A' (correct)
      * ContentImage with blue dominant pixel (b > r)  ->  perception  ->  'B' (wrong)
      * ContentImage with orange dominant (r > b), large (w > 2)  ->  knowledge full/75%  ->  'A'
      * ContentImage with orange dominant (r > b), small (w <= 2) ->  knowledge pert50/25% ->  'B'

Reports land under /tmp/evalscope_probe_demo/ so they do not pollute the repo.

Usage (from repo root):
    python -m evalscope_ext.tools.mmmu_probe_demo

Requires: Pillow  (pip install Pillow)
"""
import base64
import io
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

WORK_DIR = Path('/tmp/evalscope_probe_demo')
SIGNAL_WORK_DIR = Path('/tmp/evalscope_probe_demo_signal')
SUBSET = 'Biology'
N_RECORDS = 12
PRUNE_RATIO = 0.5

# Imports below are after sys.path setup on line 26; E402 is intentional.
from evalscope.utils.io_utils import gen_hash, safe_filename  # noqa: E402
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR  # noqa: E402
from evalscope.config import TaskConfig, EvalType  # noqa: E402

_DATASET_ID = 'AI-ModelScope/MMMU'
_SPLIT = 'validation'
_SUBSET = SUBSET

_hash = gen_hash(f'{_DATASET_ID}{_SPLIT}{_SUBSET}None{{}}')
_probe_cfg = TaskConfig(datasets=['mmmu_pruned'], eval_type=EvalType.MOCK_LLM)
_base = (_probe_cfg.dataset_dir or DEFAULT_EVALSCOPE_CACHE_DIR)
_cache_dir = Path(_base) / 'datasets' / f'{safe_filename(_DATASET_ID)}-{_hash}'

print(f'[probe_demo] Biology subset cache path: {_cache_dir}')


def _make_tiny_png(r: int, g: int, b: int) -> bytes:
    """Return PNG bytes for a 4x4 solid-colour image."""
    from PIL import Image as PILImage
    img = PILImage.new('RGB', (4, 4), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _build_synthetic_dataset(cache_dir: Path) -> None:
    """Create a 12-record synthetic MMMU/Biology dataset and save it to cache_dir."""
    import datasets as hf_datasets

    print(f'[probe_demo] Creating {N_RECORDS}-record synthetic MMMU/{SUBSET} dataset...')

    perception_png = _make_tiny_png(0, 0, 200)     # blue  -> perception (Tables/Diagrams)
    knowledge_png = _make_tiny_png(200, 100, 0)     # orange -> knowledge (Photographs/Paintings)

    IMG_TYPES_PERCEPTION = ["['Tables']", "['Diagrams']", "['Plots and Charts']"]
    IMG_TYPES_KNOWLEDGE = ["['Photographs']", "['Paintings']", "['Portraits']"]
    DIFFICULTIES = ['easy', 'medium', 'hard']
    CHOICES = "['Option A', 'Option B', 'Option C', 'Option D']"

    records = []
    for i in range(N_RECORDS):
        is_perception = i % 2 == 0
        img_type = IMG_TYPES_PERCEPTION[i // 2 % 3] if is_perception else IMG_TYPES_KNOWLEDGE[i // 2 % 3]
        png_bytes = perception_png if is_perception else knowledge_png

        records.append({
            'id': f'{SUBSET}_val_{i}',
            'question': f'Question {i}: What does <image 1> illustrate?',
            'question_type': 'multiple-choice',
            'options': CHOICES,
            'answer': 'A',
            'image_1': {'bytes': png_bytes, 'path': None},
            'image_2': None,
            'image_3': None,
            'image_4': None,
            'image_5': None,
            'image_6': None,
            'image_7': None,
            'img_type': img_type,
            'topic_difficulty': DIFFICULTIES[i % 3],
            'subfield': 'Cell Biology',
            'explanation': f'Test explanation for sample {i}.',
        })

    ds = hf_datasets.Dataset.from_list(records)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    print(f'[probe_demo] Saved {len(records)} synthetic records -> {cache_dir}')


def _make_probe_cfg(work_dir: Path) -> TaskConfig:
    return TaskConfig(
        model='mockllm',
        eval_type=EvalType.MOCK_LLM,
        datasets=['mmmu_pruned'],
        dataset_args={
            'mmmu_pruned': {
                'subset_list': [SUBSET],
                'prune_ratio': PRUNE_RATIO,
                'probe_mode': 'both',
                'probe_perturb_type': 'scale',
            }
        },
        work_dir=str(work_dir),
        no_timestamp=True,
    )


def _print_variant_table(review_files):
    all_records = []
    for rf in review_files:
        for line in rf.read_text().splitlines():
            if line.strip():
                all_records.append(json.loads(line))
    counts = {}
    for rec in all_records:
        meta = (rec.get('sample_score') or {}).get('sample_metadata') or {}
        key = (meta.get('img_group', '?'), meta.get('variant', '?'))
        counts[key] = counts.get(key, 0) + 1
    print(f'  {"img_group":<14}  {"variant":<16}  {"count":>6}')
    print(f'  {"-" * 40}')
    for (g, v), n in sorted(counts.items()):
        print(f'  {g:<14}  {v:<16}  {n:>6}')


if not _cache_dir.exists():
    _build_synthetic_dataset(_cache_dir)
else:
    print(f'[probe_demo] Using existing cached dataset at {_cache_dir}')

import logging  # noqa: E402
logging.getLogger('evalscope').setLevel(logging.INFO)

from evalscope.run import run_task  # noqa: E402
from evalscope_ext.tools.mmmu_probe_report import run_report  # noqa: E402

# ---------------------------------------------------------------------------
# Phase 1: Standard MockLLM (all-zeros baseline, proves plumbing)
# ---------------------------------------------------------------------------

print(f'\n{"=" * 60}')
print('[probe_demo] Phase 1: Standard MockLLM (plumbing verification)')
print(f'  prune_ratio  : {PRUNE_RATIO}  probe_mode: both  perturb: scale')
print(f'  work_dir     : {WORK_DIR}')
print('=' * 60)

shutil.rmtree(WORK_DIR, ignore_errors=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
run_task(_make_probe_cfg(WORK_DIR))

review_files = sorted(WORK_DIR.glob('reviews/**/*.jsonl'))
print(f'\n[probe_demo] Variants produced (6 base samples x 5 variants = 30):')
_print_variant_table(review_files)

print(f'\n[probe_demo] Phase 1 probe report (all zeros expected):')
report1 = run_report(WORK_DIR)

# ---------------------------------------------------------------------------
# Phase 2: Demo signal (encoder-weakness simulation)
# ---------------------------------------------------------------------------

print(f'\n{"=" * 60}')
print('[probe_demo] Phase 2: Demo signal -- encoder weakness simulation')
print('  Signal logic (content-aware MockLLM):')
print('    text_only          -> "A" (correct)  for all groups')
print('    perception + image -> "B" (wrong)     encoder weak at any resolution')
print('    knowledge  + image/pert75 (4x4/3x3) -> "A" (correct)')
print('    knowledge  + pert50/pert25 (2x2/1x1) -> "B" (wrong)')
print(f'  work_dir     : {SIGNAL_WORK_DIR}')
print('=' * 60)

import evalscope.models.mockllm as _mockllm_mod  # noqa: E402
from evalscope.api.model.model_output import ModelOutput  # noqa: E402

_original_generate = _mockllm_mod.MockLLM.generate


def _signal_generate(self, input, tools, tool_choice, config):
    """Content-aware MockLLM.generate for encoder-weakness simulation."""
    from evalscope.api.messages.content import ContentImage

    content_items = []
    for msg in input:
        content = getattr(msg, 'content', None)
        if isinstance(content, list):
            content_items.extend(content)

    # Detect text_only variant: placeholder "[IMAGE: ..." present, no ContentImage
    all_text = ' '.join(
        getattr(item, 'text', '')
        for item in content_items
        if not isinstance(item, ContentImage)
    )
    if '[IMAGE:' in all_text:
        return ModelOutput.from_content(model='mockllm', content='A')

    img_b64 = next(
        (item.image for item in content_items if isinstance(item, ContentImage)),
        None,
    )
    if img_b64 is None:
        return ModelOutput.from_content(model='mockllm', content='B')

    # Decode PNG to get pixel color (group) and dimensions (degradation level)
    try:
        from PIL import Image as PILImage
        raw = img_b64.split(',', 1)[1] if ',' in img_b64 else img_b64
        img = PILImage.open(io.BytesIO(base64.b64decode(raw)))
        w, h = img.size
        r, g, b = img.getpixel((0, 0))[:3]  # top-left pixel

        # Blue dominant (b > r) -> perception group
        # Orange dominant (r > b) -> knowledge group
        is_perception = b > r

        if is_perception:
            # Perception encoder is weak: wrong at every resolution
            return ModelOutput.from_content(model='mockllm', content='B')
        else:
            # Knowledge encoder: works at full/75% (w > 2), fails at 50%/25% (w <= 2)
            if w > 2:
                return ModelOutput.from_content(model='mockllm', content='A')
            else:
                return ModelOutput.from_content(model='mockllm', content='B')

    except Exception:
        # Any decode failure: treat as wrong (graceful)
        return ModelOutput.from_content(model='mockllm', content='B')


# Patch MockLLM for the duration of the signal run
_mockllm_mod.MockLLM.generate = _signal_generate

try:
    shutil.rmtree(SIGNAL_WORK_DIR, ignore_errors=True)
    SIGNAL_WORK_DIR.mkdir(parents=True, exist_ok=True)
    run_task(_make_probe_cfg(SIGNAL_WORK_DIR))
finally:
    _mockllm_mod.MockLLM.generate = _original_generate

print(f'\n[probe_demo] Phase 2 probe report (non-trivial metrics expected):')
report2 = run_report(SIGNAL_WORK_DIR)

W = 70
print(f'\n{"=" * W}')
print('  SUMMARY: Phase 1 (all-zeros) vs Phase 2 (signal)')
print(f'{"=" * W}')
print(f'\n  encoder_gap = acc(text_only) - acc(image)')
print(f'  {"group":<14}  {"phase1 gap":>12}  {"phase2 gap":>12}  {"interpretation"}')
print(f'  {"-" * 62}')

eg1 = report1['encoder_gap']
eg2 = report2['encoder_gap']
for group in sorted(set(list(eg1.keys()) + list(eg2.keys()))):
    g1 = eg1.get(group)
    g2 = eg2.get(group)
    f1 = f'{g1:.4f}' if g1 is not None else '  n/a  '
    f2 = f'{g2:.4f}' if g2 is not None else '  n/a  '
    if group == 'perception' and g2 is not None and g2 > 0:
        note = '<-- encoder weak (gap > 0)'
    elif group == 'knowledge' and g2 is not None and g2 == 0:
        note = 'encoder fine at full res'
    else:
        note = ''
    print(f'  {group:<14}  {f1:>12}  {f2:>12}  {note}')

print(f'\n  degradation curve (phase 2):')
print(f'  {"group":<14}  {"image":>8}  {"pert75":>8}  {"pert50":>8}  {"pert25":>8}  {"interpretation"}')
print(f'  {"-" * 70}')

dc2 = report2['degradation_curve']
for group in sorted(dc2.keys()):
    curve = dc2[group]
    img_v = curve.get('image')
    p75 = curve.get('pert_scale75')
    p50 = curve.get('pert_scale50')
    p25 = curve.get('pert_scale25')

    def f(v):
        return f'{v:.4f}' if v is not None else '  n/a  '

    if group == 'knowledge' and img_v and img_v > 0 and p50 is not None and p50 == 0:
        note = '<-- curve declines at pert50'
    elif group == 'perception' and img_v is not None and img_v == 0:
        note = 'flat zero (encoder uniformly weak)'
    else:
        note = ''
    print(f'  {group:<14}  {f(img_v):>8}  {f(p75):>8}  {f(p50):>8}  {f(p25):>8}  {note}')

print(f'\n{"=" * W}')
print('\n[probe_demo] Completed successfully.')
print("""
Reviewer commands (substitute a real model):

  # Probe run with real model
  evalscope eval \\
    --model Qwen/Qwen2.5-VL-7B-Instruct \\
    --api-url http://localhost:8000/v1/chat/completions \\
    --api-key EMPTY \\
    --datasets mmmu_pruned \\
    --dataset-args '{"prune_ratio":0.2,"probe_mode":"both","probe_perturb_type":"scale"}' \\
    --work-dir ./results_probe --no-timestamp

  # Probe report
  python -m evalscope_ext.tools.mmmu_probe_report --work-dir /tmp/evalscope_probe_demo_signal
""")
