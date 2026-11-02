import os

import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

CLASS_TO_PIECE = {
    "black-bishop": chess.Piece(chess.BISHOP, chess.BLACK),
    "black-king": chess.Piece(chess.KING, chess.BLACK),
    "black-knight": chess.Piece(chess.KNIGHT, chess.BLACK),
    "black-pawn": chess.Piece(chess.PAWN, chess.BLACK),
    "black-queen": chess.Piece(chess.QUEEN, chess.BLACK),
    "black-rook": chess.Piece(chess.ROOK, chess.BLACK),
    "white-bishop": chess.Piece(chess.BISHOP, chess.WHITE),
    "white-king": chess.Piece(chess.KING, chess.WHITE),
    "white-knight": chess.Piece(chess.KNIGHT, chess.WHITE),
    "white-pawn": chess.Piece(chess.PAWN, chess.WHITE),
    "white-queen": chess.Piece(chess.QUEEN, chess.WHITE),
    "white-rook": chess.Piece(chess.ROOK, chess.WHITE),
}
PIECE_TO_CLASS = {v: k for k, v in CLASS_TO_PIECE.items()}
CLASSES = list(CLASS_TO_PIECE.keys())
CORNERS = ["h1", "a1", "a8", "h8"]
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE
ABBR_MAP = {
    "black-pawn": "BP",
    "white-pawn": "WP",
    "black-knight": "BN",
    "white-knight": "WN",
    "black-bishop": "BB",
    "white-bishop": "WB",
    "black-rook": "BR",
    "white-rook": "WR",
    "black-king": "BK",
    "white-king": "WK",
    "black-queen": "BQ",
    "white-queen": "WQ",
}
CHAR_TO_CATEGORY = {
    "k": "black-king",
    "q": "black-queen",
    "r": "black-rook",
    "p": "black-pawn",
    "b": "black-bishop",
    "n": "black-knight",
    "K": "white-king",
    "Q": "white-queen",
    "R": "white-rook",
    "P": "white-pawn",
    "B": "white-bishop",
    "N": "white-knight",
}
