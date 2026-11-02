"""Predict a board for every pieces image with ChessQ Lite, to orient and audit the converted labels.

uv run python -m camera_chess.labels.predict_chessq --checkpoint checkpoints/lite/cq-vit-s-644-v4-seed2-safetensors-fp16-r2.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from camera_chess.chessqueries.models.predictor import Predictor
from camera_chess.constants import LABELS_DIR

# Kept in step with make_chessqueries_dataset.py, which needs OpenCV rather than torch
SYMBOLS = "bknpqrBKNPQR"
EXCLUDED_PREFIXES = ("chessred2k",)


def placement_to_labels(placement: str) -> list[int]:
    """64 class indices (-1 for empty) in FEN order, a8..h1"""
    labels = []
    for ch in placement.replace("/", ""):
        labels.extend([-1] * int(ch) if ch.isdigit() else [SYMBOLS.index(ch)])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(LABELS_DIR))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(LABELS_DIR) / "chessq_predictions.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(args.checkpoint, device="cuda")
    auto_path = args.root / "xcorners_auto.json"
    auto = set(json.loads(auto_path.read_text())) if auto_path.exists() else set()
    predictions = json.loads(args.out.read_text()) if args.out.exists() else {}
    # Every image the converter can place a grid on, by hand-labelled or model-found corners
    paths = [
        p
        for split in ("train", "val")
        for p in sorted((args.root / "pieces" / split / "images").glob("*.jpg"))
        if not p.name.startswith(EXCLUDED_PREFIXES)
        and p.stem not in predictions
        and ((args.root / "xcorners" / split / "labels" / f"{p.stem}.txt").exists() or f"{split}/{p.stem}" in auto)
    ]

    for start in tqdm(range(0, len(paths), args.batch_size), unit="batch"):
        batch = paths[start : start + args.batch_size]
        for path, pred in zip(batch, predictor.predict(batch, batch_size=args.batch_size)):
            predictions[path.stem] = placement_to_labels(pred.board.placement)
    args.out.write_text(json.dumps(predictions))
    logger.info(f"{len(predictions)} predictions in {args.out}")


if __name__ == "__main__":
    main()
