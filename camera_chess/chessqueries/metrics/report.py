"""Shared eval-script reporting: the metric printout and the metrics/preds JSON
pair every eval entry point writes (consumed by the visualizer and the paper
tooling), plus the CLI split resolution they all repeat."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from camera_chess.chessqueries.core import Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset


def eval_tag(dataset_value: str, split: Split | None, sep: str = "_") -> str:
    """The ``<dataset>_<split>`` slug that names metrics/preds files."""
    return f"{dataset_value}{sep}{split.value if split else 'all'}"


def resolve_split(ds_obj: ChessDataset, split_arg: str | None) -> Split | None:
    """CLI boundary: the dataset's `Split` for a ``--split`` string, or None for
    split-less datasets. Exits with a usage message when a required split is missing."""
    split = Split(split_arg) if (ds_obj.splits and split_arg) else None
    if ds_obj.splits and split is None:
        raise SystemExit(f"{ds_obj.name.value} requires --split {[s.value for s in ds_obj.splits]}")
    return split


def print_metrics(metrics: dict, header: str) -> None:
    rows = (f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}" for k, v in metrics.items())
    logger.info("\n".join([f"=== {header} ===", *rows]))


def print_evaluation_inventory(inventory: dict) -> None:
    """Make diagnostic partial runs unmistakable in logs."""
    split = inventory["split"] or "all"
    mode = inventory["mode"].upper()
    status = "PAPER-COMPARABLE" if inventory["data_complete"] else "NON-PAPER"
    logger.info(
        f"data {inventory['dataset']}[{split}] [{mode}; {status}]: "
        f"expected={inventory['expected_samples']}  "
        f"labelled={inventory['labelled_samples']}  "
        f"available={inventory['available_samples']}"
    )


def write_eval_outputs(
    out_dir: Path,
    tag: str,
    metrics: dict,
    samples: list[BoardSample],
    preds: list[list[int]],
) -> Path:
    """Write ``metrics_<tag>.json`` + ``preds_<tag>.json`` (sample_id -> 64 labels)."""
    pairs = list(zip(samples, preds, strict=True))
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for sample, _ in pairs:
        if sample.sample_id in seen_ids:
            duplicate_ids.add(sample.sample_id)
        seen_ids.add(sample.sample_id)
    if duplicate_ids:
        raise ValueError(f"duplicate sample IDs: {sorted(duplicate_ids)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    predictions = {sample.sample_id: pred for sample, pred in pairs}
    (out_dir / f"preds_{tag}.json").write_text(json.dumps(predictions))
    return out_dir
