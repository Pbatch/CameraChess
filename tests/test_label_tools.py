import numpy as np

from camera_chess.labels import frames, predict_chessq
from camera_chess.labels.label_orientation import board_corners
from camera_chess.labels.review_test_labels import fen_squares, squares_fen

FEN = "r3k2r/pp3ppp/8/3pP3/8/8/PPP2PPP/R3K2R"


def test_fen_squares_round_trip():
    squares = fen_squares(FEN)
    assert len(squares) == 64 and squares[:5] == ["r", "", "", "", "k"]
    assert squares_fen(squares) == FEN
    assert squares_fen([""] * 64) == "8/8/8/8/8/8/8/8"


def test_predictions_use_the_frames_class_indices():
    assert predict_chessq.SYMBOLS == frames.SYMBOLS
    assert predict_chessq.EXCLUDED_PREFIXES == frames.EXCLUDED_PREFIXES
    squares = fen_squares(FEN)
    indices = [i for i, s in enumerate(squares) if s]
    squares_fr = np.array([[i % 8, 7 - i // 8] for i in indices])
    classes = np.array([frames.SYMBOLS.index(squares[i]) for i in indices])
    assert predict_chessq.placement_to_labels(FEN) == frames.to_labels(squares_fr, classes).tolist()


def test_board_corners_turn_with_the_rotation():
    corners = board_corners(np.eye(3))
    # Unrotated, a1's outer corner is the lattice's (-1, -1), and h8's is (7, 7)
    assert corners[0] == {"a1": [-1, -1], "h1": [7, -1], "h8": [7, 7], "a8": [-1, 7]}
    # Each rotation moves a1 to another corner, and back after four
    assert len({tuple(c["a1"]) for c in corners}) == 4
    assert corners[2]["a1"] == corners[0]["h8"]
