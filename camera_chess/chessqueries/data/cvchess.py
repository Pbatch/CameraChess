"""CVChess dataset: images plus FENs recovered from the vendored labels."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config

RESOURCES = Path(__file__).parent / "resources"
# The labels are one game in order: each position photographed from the same 4 viewpoints, and
# neighbouring positions a move apart. So splits take whole runs of positions, never single photos
POSITIONS_PER_CHUNK = 4
SPLIT_SHARES = {Split.TRAIN: 0.7, Split.VAL: 0.15, Split.TEST: 0.15}


class Viewpoint(str, Enum):
    """Where the WHITE player sits in the photo -- a single cell of a 3x3 spatial
    grid, so board corners are valid answers. ``BOTTOM`` is the standard
    white-player view; the others see the board obliquely or from Black's side."""

    TOP_LEFT = "top_left"
    TOP = "top"
    TOP_RIGHT = "top_right"
    LEFT = "left"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom_right"


def load_labels() -> list[dict]:
    return json.loads((RESOURCES / "cvchess_labels.json").read_text())


def load_viewpoints() -> dict[str, Viewpoint | None]:
    """sample_id -> camera viewpoint (where the white player sits), if labelled."""
    raw = json.loads((RESOURCES / "cvchess_viewpoints.json").read_text())["white_side"]
    return {sid: Viewpoint(v) if v else None for sid, v in raw.items()}


def assign_splits(labels: list[dict]) -> list[Split]:
    """Split per label: runs of POSITIONS_PER_CHUNK positions, in game order, each to the split
    furthest below its share (as Fenit and Samryan assign their games), so every phase of the game
    is in each split"""
    position = 0
    chunks: list[list[int]] = []
    for i, label in enumerate(labels):
        if i and label["gt_fen"] != labels[i - 1]["gt_fen"]:
            position += 1
        if position // POSITIONS_PER_CHUNK == len(chunks):
            chunks.append([])
        chunks[-1].append(i)
    counts = dict.fromkeys(SPLIT_SHARES, 0)
    out: list[Split] = [Split.TRAIN] * len(labels)
    for chunk in chunks:
        split = min(SPLIT_SHARES, key=lambda s: (counts[s] - SPLIT_SHARES[s] * len(labels)) / SPLIT_SHARES[s])
        counts[split] += len(chunk)
        for i in chunk:
            out[i] = split
    return out


class CVChess(ChessDataset):
    """Only the images live under ``root/images``; the labels are vendored in ``resources``."""

    name = DatasetName.CVCHESS
    # Not an official split: see assign_splits
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 224, Split.VAL: 64, Split.TEST: 64})

    @staticmethod
    def default_root() -> Path:
        return get_config().cvchess_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        viewpoints = load_viewpoints()
        labels = load_labels()
        samples: list[BoardSample] = []
        for entry, entry_split in zip(labels, assign_splits(labels), strict=True):
            if entry_split is not split:
                continue
            sample_id = Path(entry["image"]).stem
            samples.append(
                BoardSample(
                    image_path=self.root / "images" / entry["image"],
                    board=Board.from_fen(entry["gt_fen"]),
                    dataset=self.name,
                    sample_id=sample_id,
                    split=split,
                    meta={"gt_fen": entry["gt_fen"], "white_side": viewpoints.get(sample_id)},
                )
            )
        return samples
