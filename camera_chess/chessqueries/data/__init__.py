"""Dataset loaders. Every dataset subclasses :class:`ChessDataset` and yields a
common :class:`BoardSample`. Importing this package registers all datasets.
"""

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
