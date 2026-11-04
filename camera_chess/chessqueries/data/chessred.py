"""ChessReD dataset (4TU.ResearchData): real smartphone photos.

We ship one set of images: the authors' offline-resized 1024x1024 set — the
exact inputs the released checkpoint was trained on, and what reproduces the
paper (the raw 3072x3072 images + a runtime `Resize` do not; see upstream issue
tmasouris/end-to-end-chess-recognition#5)."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from camera_chess.chessqueries.core import Board, Piece, Split, Square
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config

# ChessReD category names -> Piece.
CATEGORY_TO_PIECE: dict[str, Piece] = {
    "empty": Piece.EMPTY,
    "white-pawn": Piece.WHITE_PAWN,
    "white-knight": Piece.WHITE_KNIGHT,
    "white-bishop": Piece.WHITE_BISHOP,
    "white-rook": Piece.WHITE_ROOK,
    "white-queen": Piece.WHITE_QUEEN,
    "white-king": Piece.WHITE_KING,
    "black-pawn": Piece.BLACK_PAWN,
    "black-knight": Piece.BLACK_KNIGHT,
    "black-bishop": Piece.BLACK_BISHOP,
    "black-rook": Piece.BLACK_ROOK,
    "black-queen": Piece.BLACK_QUEEN,
    "black-king": Piece.BLACK_KING,
}


class ChessReD(ChessDataset):
    name = DatasetName.CHESSRED
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 6_479, Split.VAL: 2_192, Split.TEST: 2_129})

    @staticmethod
    def default_root() -> Path:
        return get_config().chessred_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        assert split is not None  # _validate_split requires one
        path = self.root / "annotations.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Download ChessReD first:\n    uv run python scripts/download_data.py chessred"
            )
        ann = json.loads(path.read_text())
        cat_name = {c["id"]: c["name"] for c in ann["categories"]}
        img_path = {im["id"]: im["path"] for im in ann["images"]}

        boards = {img_id: [Piece.EMPTY] * 64 for img_id in ann["splits"][split.value]["image_ids"]}
        for piece in ann["annotations"]["pieces"]:
            p = CATEGORY_TO_PIECE[cat_name[piece["category_id"]]]
            if piece["image_id"] in boards and not p.is_empty:
                boards[piece["image_id"]][Square.from_name(piece["chessboard_position"]).index] = p

        return [
            BoardSample(
                image_path=self.root / img_path[img_id],
                board=Board(tuple(pieces)),
                dataset=self.name,
                sample_id=str(img_id),
                split=split,
            )
            for img_id, pieces in boards.items()
        ]
