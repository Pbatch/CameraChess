"""Dataset loaders. Every dataset subclasses :class:`ChessDataset` and yields a
common :class:`BoardSample`.

Loaders are imported on first use, by :func:`get_dataset` or by name from this package, so a
job only needs the dependencies of the datasets it reads.
"""

import importlib
from typing import TYPE_CHECKING

from camera_chess.chessqueries.data.base import (
    DATASET_REGISTRY,
    BoardImageDataset,
    BoardSample,
    ChessDataset,
    DatasetCompleteness,
    DatasetIncompleteError,
    DatasetName,
    get_dataset,
)

if TYPE_CHECKING:
    from camera_chess.chessqueries.data.chesscam import ChessCam
    from camera_chess.chessqueries.data.chesscog import ChessCog
    from camera_chess.chessqueries.data.chessred import ChessReD
    from camera_chess.chessqueries.data.cvchess import CVChess, Viewpoint
    from camera_chess.chessqueries.data.digitizer import Digitizer
    from camera_chess.chessqueries.data.fenit import Fenit
    from camera_chess.chessqueries.data.samryan import Samryan
    from camera_chess.chessqueries.data.slcc import SLCC
    from camera_chess.chessqueries.data.synthetic import Synthetic
    from camera_chess.chessqueries.data.yunker import Yunker

# Exported name -> module that defines it
_LAZY = {
    "ChessCam": "chesscam",
    "ChessCog": "chesscog",
    "ChessReD": "chessred",
    "CVChess": "cvchess",
    "Viewpoint": "cvchess",
    "Digitizer": "digitizer",
    "Fenit": "fenit",
    "Samryan": "samryan",
    "SLCC": "slcc",
    "Synthetic": "synthetic",
    "Yunker": "yunker",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


__all__ = [
    "DATASET_REGISTRY",
    "SLCC",
    "BoardImageDataset",
    "BoardSample",
    "CVChess",
    "ChessCam",
    "ChessCog",
    "ChessDataset",
    "ChessReD",
    "DatasetCompleteness",
    "DatasetIncompleteError",
    "DatasetName",
    "Digitizer",
    "Fenit",
    "Samryan",
    "Synthetic",
    "Viewpoint",
    "Yunker",
    "get_dataset",
]
