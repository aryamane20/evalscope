import ast
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter, SUBSET_LIST
from evalscope.benchmarks.pruned.mixin import PruningMixin
from evalscope.constants import Tags

IMG_TYPE_GROUPS: Dict[str, str] = {
    'Tables': 'perception',
    'Plots and Charts': 'perception',
    'Diagrams': 'perception',
    'Chemical Structures': 'perception',
    'Microscopic Images': 'perception',
    'Medical Images': 'perception',
    'Pathological Images': 'perception',
    'Technical Blueprints': 'perception',
    'Geometric Shapes': 'perception',
    'Body Scans: MRI, CT scans, and X-rays': 'perception',
    'Maps': 'perception',
    'Trees and Graphs': 'perception',
    'Mathematical Notations': 'perception',
    'DNA Sequences': 'perception',
    'Screenshots': 'perception',
    'Photographs': 'knowledge',
    'Paintings': 'knowledge',
    'Comics and Cartoons': 'knowledge',
    'Portraits': 'knowledge',
    'Sculpture': 'knowledge',
    'Landscapes': 'knowledge',
    'Sketches and Drafts': 'knowledge',
    'Logos and Branding': 'knowledge',
    'Poster': 'knowledge',
    'Advertisements': 'knowledge',
    'Icons and Symbols': 'knowledge',
    'Other': 'knowledge',
}

TOPIC_DIFFICULTY_MAP: Dict[str, float] = {
    'easy': 0.2,
    'medium': 0.5,
    'hard': 0.8,
    '1': 0.1, '2': 0.3, '3': 0.5, '4': 0.7, '5': 0.9,
}


def parse_img_types(img_type_raw) -> List[str]:
    """Parse img_type from any format the MMMU dataset emits.

    The MMMU HuggingFace dataset stores img_type as a Python-list string
    (e.g. ``"['Tables']"``). Handles a list string, an actual list, or a plain
    string.
    """
    if isinstance(img_type_raw, list):
        return [str(t).strip() for t in img_type_raw]
    if isinstance(img_type_raw, str):
        if img_type_raw.startswith('['):
            try:
                parsed = ast.literal_eval(img_type_raw)
                return [str(t).strip() for t in parsed]
            except (ValueError, SyntaxError):
                pass
        return [img_type_raw.strip()] if img_type_raw.strip() else []
    return []


def img_type_to_group(img_type_raw) -> str:
    """Map raw img_type to 'perception' or 'knowledge'.

    A sample tagged with any perception type is treated as perception so hard
    visual samples are never dropped.
    """
    types = parse_img_types(img_type_raw)
    for t in types:
        if IMG_TYPE_GROUPS.get(t, 'knowledge') == 'perception':
            return 'perception'
    return 'knowledge'


def difficulty_to_float(td) -> float:
    """Map topic_difficulty to a float in [0, 1]."""
    if isinstance(td, (int, float)):
        v = float(td)
        return v / 10.0 if v > 1.0 else v
    key = str(td).strip().lower()
    return TOPIC_DIFFICULTY_MAP.get(key, 0.5)


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (pruned)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

A compact, representative subset of MMMU (Massive Multi-discipline Multimodal
Understanding) produced by the pruning engine. The subset preserves both the
subject distribution and the perception/knowledge image-type balance of the
full benchmark.

## Feature source (live, no shipped files needed)

Buckets are derived at load time from two metadata fields on each live Sample:

  - ``sample.metadata['img_type']``   coarsened to "perception" or "knowledge"
    via ``IMG_TYPE_GROUPS`` in ``mmmu_pruned_adapter.py`` (explicit, editable dict)
  - ``subset_name``                   academic subject (e.g. "Biology")

Bucket: ``f"{subject}__{img_group}"``, e.g. ``"Biology__perception"``.

Difficulty signal: ``sample.metadata['topic_difficulty']`` mapped to [0, 1]
via ``TOPIC_DIFFICULTY_MAP``.

No shipped prediction or review file is read at eval time.

## Editing the perception/knowledge mapping

Open ``evalscope/benchmarks/pruned/mmmu_pruned_adapter.py`` and edit
``IMG_TYPE_GROUPS``. All 15 perception types and 12 knowledge types from the
MMMU validation split are listed there.

## Image encoder probe modes

Set ``probe_mode`` (flat or nested form) to isolate the image encoder ("eyes")
from the reasoning LLM ("brain"):

  ``text_control``   For each pruned sample emits a paired text-only variant
                     where every image is replaced by ``[IMAGE: <label>]``.
                     Metric: encoder_gap = acc(text_only) - acc(image).
                     Large positive gap means the model can reason but cannot
                     see, revealing encoder weakness.

  ``perturbation``   For each pruned sample emits degraded-image variants
                     (scale75/50/25 or blur1/2/4 depending on
                     ``probe_perturb_type``).
                     Metric: accuracy degradation curve per group.
                     Graceful (stable) = strong encoder; collapse = weak.

  ``both``           Combines both controls.

  ``none``           Default -- standard pruned evaluation, no variants.

## Reviewer commands

    # Full eval
    evalscope eval --model <m> --datasets mmmu --output ./results_full/

    # Pruned eval (flat form, matches the spec)
    evalscope eval --model <m> --datasets mmmu_pruned \\
        --dataset-args '{"pruning_strategy":"stratified_coreset","prune_ratio":0.1}' \\
        --output ./results_pruned/

    # Probe run (both controls, scale perturbation)
    evalscope eval --model <m> --datasets mmmu_pruned \\
        --dataset-args '{"prune_ratio":0.2,"probe_mode":"both","probe_perturb_type":"scale"}' \\
        --output ./results_probe/

    # Probe report
    python -m evalscope_ext.tools.mmmu_probe_report --work-dir ./results_probe/

    # Compare
    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ --pruned ./results_pruned/ --tolerance 0.05
""",
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy name.',
                'value': 'stratified_coreset',
                'choices': ['stratified', 'coreset', 'difficulty_spread', 'stratified_coreset'],
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to KEEP (0 < prune_ratio <= 1).',
                'value': 0.1,
            },
            'seed': {
                'type': 'int',
                'description': 'Random seed for deterministic subset selection.',
                'value': 42,
            },
            'n_diff_buckets': {
                'type': 'int',
                'description': 'Number of quantile buckets for topic_difficulty stratification.',
                'value': 3,
            },
            'probe_mode': {
                'type': 'str',
                'description': (
                    'Image encoder probe mode. "text_control" emits paired text-only variants; '
                    '"perturbation" emits degraded-image variants; "both" combines both; '
                    '"none" disables probing (standard pruned eval).'
                ),
                'value': 'none',
                'choices': ['none', 'text_control', 'perturbation', 'both'],
            },
            'probe_perturb_type': {
                'type': 'str',
                'description': (
                    'Perturbation type when probe_mode includes perturbation. '
                    '"scale" downscales to 75/50/25 percent; '
                    '"blur" applies Gaussian blur at radius 1/2/4.'
                ),
                'value': 'scale',
                'choices': ['scale', 'blur'],
            },
        },
    )
)
class MMMUPrunedAdapter(PruningMixin, MMMUAdapter):
    """MMMU with subject + image-type stratified pruning and optional encoder probing.

    Inherits multimodal input handling and scoring from MMMUAdapter.
    PruningMixin.load_dataset() calls the parent chain for full loading, then
    prunes preserving (subject x img_group) proportions.

    When probe_mode != 'none', the pruned samples are expanded into tagged
    variants at load time so the evaluator scores them independently.  The
    'variant' and 'img_group' keys are stored in each Sample's metadata and
    flow through to the review JSONL for the probe reporter to read.
    """

    probe_mode: str = 'none'
    probe_perturb_type: str = 'scale'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_pruning_params(
            default_strategy='stratified_coreset',
            default_ratio=0.1,
            default_seed=42,
            default_buckets=3,
        )
        self._init_probe_params()

    def _init_probe_params(self) -> None:
        """Resolve probe parameters from the same flat+nested contracts as pruning params."""
        ep = self.extra_params or {}

        flat: Dict[str, Any] = {}
        task_config = getattr(self, '_task_config', None)
        if task_config is not None:
            dataset_args = getattr(task_config, 'dataset_args', None) or {}
            flat = dataset_args.get(self.name, {}) or {}
        nested = flat.get('extra_params', {}) or {}

        def pick(default: Any, *keys: str) -> Any:
            for key in keys:
                if key in nested:
                    return nested[key]
            for key in keys:
                if key in flat:
                    return flat[key]
            for key in keys:
                if key in ep:
                    return ep[key]
            return default

        self.probe_mode = str(pick('none', 'probe_mode'))
        self.probe_perturb_type = str(pick('scale', 'probe_perturb_type'))

    def load_dataset(self):  # type: ignore[override]
        """Load and prune via parent chain, then expand probe variants if requested."""
        pruned_dict = super().load_dataset()
        if self.probe_mode == 'none':
            return pruned_dict
        return self._expand_probe_variants(pruned_dict)

    def _extract_raw(self, sample, subset_name: str) -> Dict[str, Any]:
        """Bucket by (subject x image_type_group); difficulty from topic_difficulty.

        Fields used:
          ``sample.metadata['img_type']``         parsed via parse_img_types(),
                                                  looked up in IMG_TYPE_GROUPS
          ``subset_name``                          academic subject
          ``sample.metadata['topic_difficulty']``  mapped to [0, 1]
        """
        metadata = sample.metadata or {}
        img_type_raw = metadata.get('img_type', '')
        topic_difficulty = metadata.get('topic_difficulty', 'medium')

        group = img_type_to_group(img_type_raw)
        category = f'{subset_name}__{group}'

        return {
            'category': category,
            'difficulty_value': difficulty_to_float(topic_difficulty),
        }

    # ------------------------------------------------------------------
    # Probe variant expansion
    # ------------------------------------------------------------------

    def _expand_probe_variants(self, dataset_dict):
        """Expand each pruned sample into tagged variants for encoder probing.

        Original samples receive ``metadata['variant'] = 'image'``.
        Text-only copies receive ``variant = 'text_only'``.
        Perturbation copies receive ``variant = 'pert_<level>'``
        (e.g. ``'pert_scale75'``, ``'pert_blur2'``).
        All variants also carry ``metadata['img_group']`` ('perception' or 'knowledge').
        """
        from evalscope.api.dataset import DatasetDict as DD, MemoryDataset
        from evalscope.utils.logger import get_logger
        logger = get_logger()

        do_text = self.probe_mode in {'text_control', 'both'}
        do_pert = self.probe_mode in {'perturbation', 'both'}

        expanded: Dict[str, list] = {}
        n_base = 0
        n_total = 0

        for subset_name, dataset in dataset_dict.items():
            samples = []
            for sample in dataset:
                metadata = sample.metadata or {}
                img_type_raw = metadata.get('img_type', '')
                img_group = img_type_to_group(img_type_raw)

                sample.metadata['variant'] = 'image'
                sample.metadata['img_group'] = img_group
                samples.append(sample)
                n_base += 1

                if do_text:
                    ts = self._make_text_only_variant(sample, img_type_raw, img_group)
                    if ts is not None:
                        samples.append(ts)

                if do_pert:
                    for ps in self._make_perturbation_variants(sample, img_group, logger):
                        samples.append(ps)

            expanded[subset_name] = MemoryDataset(samples, name=subset_name)
            n_total += len(samples)

        logger.info(
            f'{self.name}: probe_mode={self.probe_mode!r} perturb_type={self.probe_perturb_type!r} '
            f'-> {n_base} base samples -> {n_total} total (with variants)'
        )
        self.test_dataset = DD(expanded)  # type: ignore[attr-defined]
        return self.test_dataset

    def _make_text_only_variant(self, sample, img_type_raw: str, img_group: str):
        """Return a copy of sample with every ContentImage replaced by a text placeholder.

        Returns None if the sample input is not a list of messages (defensive).
        """
        from evalscope.api.messages.content import ContentImage, ContentText

        img_types = parse_img_types(img_type_raw)
        label = img_types[0] if img_types else 'image'
        placeholder = f'[IMAGE: {label}]'

        if not isinstance(sample.input, list):
            return None

        new_msgs = []
        for msg in sample.input:
            content = getattr(msg, 'content', None)
            if isinstance(content, list):
                new_content = [
                    ContentText(text=placeholder) if isinstance(item, ContentImage) else item
                    for item in content
                ]
                msg = msg.model_copy(update={'content': new_content})
            new_msgs.append(msg)

        meta = dict(sample.metadata)
        meta['variant'] = 'text_only'
        meta['img_group'] = img_group
        return sample.model_copy(update={'input': new_msgs, 'metadata': meta})

    def _make_perturbation_variants(self, sample, img_group: str, logger) -> list:
        """Return degraded-image copies of sample at each perturbation level.

        Skips any level where image decoding or re-encoding fails, logging a
        warning rather than crashing.
        """
        from evalscope.api.messages.content import ContentImage

        if self.probe_perturb_type == 'scale':
            levels = [
                ('scale75', 'scale', 0.75),
                ('scale50', 'scale', 0.50),
                ('scale25', 'scale', 0.25),
            ]
        else:
            levels = [
                ('blur1', 'blur', 1),
                ('blur2', 'blur', 2),
                ('blur4', 'blur', 4),
            ]

        if not isinstance(sample.input, list):
            return []

        variants = []
        for level_name, ptype, pval in levels:
            new_msgs = []
            ok = True
            for msg in sample.input:
                content = getattr(msg, 'content', None)
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, ContentImage):
                            try:
                                new_img = self._degrade_image(item.image, ptype, pval)
                                new_content.append(item.model_copy(update={'image': new_img}))
                            except Exception as exc:
                                logger.warning(
                                    f'{self.name}: image degrade failed for '
                                    f'pert_{level_name}: {exc}; skipping variant'
                                )
                                ok = False
                                break
                        else:
                            new_content.append(item)
                    if not ok:
                        break
                    msg = msg.model_copy(update={'content': new_content})
                new_msgs.append(msg)

            if ok:
                meta = dict(sample.metadata)
                meta['variant'] = f'pert_{level_name}'
                meta['img_group'] = img_group
                variants.append(
                    sample.model_copy(update={'input': new_msgs, 'metadata': meta})
                )

        return variants

    @staticmethod
    def _degrade_image(b64_str: str, perturb_type: str, level_val: float) -> str:
        """Apply a Pillow transform to a base64-encoded PNG and return the new base64 string.

        Preserves the ``data:image/...;base64,`` header if present.
        Raises ImportError if Pillow is not installed.
        """
        import base64
        import io
        try:
            from PIL import Image as PILImage, ImageFilter
        except ImportError as exc:
            raise ImportError(
                'Pillow is required for image perturbation: pip install Pillow'
            ) from exc

        if ',' in b64_str:
            header, data = b64_str.split(',', 1)
        else:
            header, data = None, b64_str

        img_bytes = base64.b64decode(data)
        img = PILImage.open(io.BytesIO(img_bytes))

        if perturb_type == 'scale':
            new_w = max(1, int(img.width * level_val))
            new_h = max(1, int(img.height * level_val))
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=level_val))

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        new_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        return f'{header},{new_b64}' if header else new_b64
