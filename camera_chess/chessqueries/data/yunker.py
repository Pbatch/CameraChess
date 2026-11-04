"""Yunker dataset: crops of the 3D board in YunkerChess blitz videos, labelled with the position the video's 2D overlay
shows, by ``camera_chess/videos/make_training_data.py``.

Every frame is for training: the games scored by CameraChessWeb's tracker replay are the test, and are never in it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config


class Yunker(ChessDataset):
    name = DatasetName.YUNKER
    splits = (Split.TRAIN,)

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root)
        annotations_path = self.root / "annotations.json"
        if not annotations_path.is_file():
            raise FileNotFoundError(
                f"{annotations_path} not found; build it with camera_chess.videos.make_training_data"
            )
        self._annotations = json.loads(annotations_path.read_text())
        # Not frozen like the paper datasets: the conversion decides the counts
        counts = Counter(a["split"] for a in self._annotations)
        self.expected_samples = {split: counts[split.value] for split in self.splits}

    @staticmethod
    def default_root() -> Path:
        return get_config().yunker_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        assert split is not None  # _validate_split requires one
        return [
            BoardSample(
                image_path=self.root / a["image"],
                board=Board.from_fen(a["fen"]),
                dataset=self.name,
                sample_id=Path(a["image"]).stem,
                split=split,
                # The board region within the crop, as [x, y, width, height]
                meta={"group": a["group"], "box": a["box"], "model_agreement": a.get("model_agreement")},
            )
            for a in self._annotations
            if a["split"] == split.value
        ]
