"""ChessCog dataset: synthetic rendered boards with per-image FEN annotations."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config


class ChessCog(ChessDataset):
    name = DatasetName.CHESSCOG
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 4_400, Split.VAL: 146, Split.TEST: 342})

    @staticmethod
    def default_root() -> Path:
        return get_config().chesscog_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        assert split is not None  # _validate_split requires one
        samples: list[BoardSample] = []
        for json_path in sorted((self.root / "render" / split.value).glob("*.json")):
            # The renders ship as PNG; local copies are converted to JPEG, packed ones also shrunk
            img_path = json_path.with_suffix(".jpg")
            if not img_path.is_file():
                img_path = json_path.with_suffix(".png")
            samples.append(
                BoardSample(
                    image_path=img_path,
                    board=Board.from_fen(json.loads(json_path.read_text())["fen"]),
                    dataset=self.name,
                    sample_id=json_path.stem,
                    split=split,
                )
            )
        return samples
