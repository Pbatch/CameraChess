"""ChessCam dataset: frames of real games labelled with piece boxes and board x-corners,
converted to FENs by ``make_chessqueries_dataset.py`` in CameraChessLabels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config


class ChessCam(ChessDataset):
    name = DatasetName.CHESSCAM
    splits = (Split.TRAIN, Split.VAL, Split.TEST)

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root)
        annotations_path = self.root / "annotations.json"
        if not annotations_path.is_file():
            raise FileNotFoundError(f"{annotations_path} not found; build it with make_chessqueries_dataset.py")
        self._annotations = json.loads(annotations_path.read_text())
        # Not frozen like the paper datasets: the conversion decides the counts
        counts = Counter(a["split"] for a in self._annotations)
        self.expected_samples = {split: counts[split.value] for split in self.splits}

    @staticmethod
    def default_root() -> Path:
        return get_config().chesscam_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        assert split is not None  # _validate_split requires one
        return [
            BoardSample(
                image_path=self.root / a["image"],
                board=Board.from_fen(a["fen"]),
                dataset=self.name,
                sample_id=Path(a["image"]).stem,
                split=split,
                meta={"group": a["group"], "model_agreement": a.get("model_agreement")},
            )
            for a in self._annotations
            if a["split"] == split.value
        ]
