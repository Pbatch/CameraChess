"""Our synthetic renders (camera_chess/synthetic/render.py): Blender boards in ChessCog's layout and label format."""

from __future__ import annotations

from pathlib import Path

from camera_chess.chessqueries.data.base import DatasetName
from camera_chess.chessqueries.data.chesscog import ChessCog
from camera_chess.config import get_config


class Synthetic(ChessCog):
    name = DatasetName.SYNTHETIC

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root)
        # Not frozen like ChessCog's: the render run decides the counts
        self.expected_samples = {
            split: sum(1 for _ in (self.root / "render" / split.value).glob("*.json")) for split in self.splits
        }

    @staticmethod
    def default_root() -> Path:
        return get_config().synthetic_root
