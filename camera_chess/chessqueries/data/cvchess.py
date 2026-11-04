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


class CVChess(ChessDataset):
    """Only the images live under ``root/images``; the labels are vendored in ``resources``."""

    name = DatasetName.CVCHESS
    splits = ()  # no official split
    expected_samples = MappingProxyType({None: 352})

    @staticmethod
    def default_root() -> Path:
        return get_config().cvchess_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        viewpoints = load_viewpoints()
        samples: list[BoardSample] = []
        for entry in load_labels():
            sample_id = Path(entry["image"]).stem
            samples.append(
                BoardSample(
                    image_path=self.root / "images" / entry["image"],
                    board=Board.from_fen(entry["gt_fen"]),
                    dataset=self.name,
                    sample_id=sample_id,
                    meta={"gt_fen": entry["gt_fen"], "white_side": viewpoints.get(sample_id)},
                )
            )
        return samples
