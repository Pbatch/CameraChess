import pytest

from camera_chess.chessqueries.core import Board, Piece, Square
from camera_chess.chessqueries.metrics import aggregate, aggregate_subsets, score_board

EMPTY = int(Piece.EMPTY)


def _start():
    return Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").labels


def test_perfect_board():
    gt = _start()
    s = score_board(gt, gt)
    assert s.exact and s.n_squares_correct == 64


def test_one_square_off():
    gt = _start()
    pred = list(gt)
    pred[Square.from_name("e1").index] = EMPTY  # drop the white king
    s = score_board(pred, gt)
    assert not s.exact
    assert s.n_wrong == 1
    assert s.within(1) and not s.within(0)


def test_occupancy_vs_piece():
    gt = _start()
    pred = list(gt)
    # Swap a white knight for a white bishop: occupancy still correct, piece wrong.
    pred[Square.from_name("b1").index] = int(Piece.WHITE_BISHOP)
    s = score_board(pred, gt)
    assert s.n_occupancy_correct == 64
    assert s.n_wrong == 1


def test_accepts_board_objects():
    gt = Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert score_board(gt, gt).exact


def test_aggregate():
    gt = _start()
    wrong = list(gt)
    wrong[0] = EMPTY
    agg = aggregate([gt, wrong], [gt, gt])
    assert agg["n_boards"] == 2
    assert agg["board_accuracy"] == 0.5
    assert agg["board_accuracy@1"] == 1.0
    assert agg["mean_wrong_squares"] == 0.5


def test_aggregate_subsets():
    gt = _start()
    wrong = list(gt)
    wrong[0] = EMPTY
    # "low" bucket: one perfect + one wrong board; "high" bucket: one perfect board.
    out = aggregate_subsets([gt, wrong, gt], [gt, gt, gt], ["low", "low", "high"])
    assert out["overall"]["n_boards"] == 3
    assert set(out["subsets"]) == {"low", "high"}
    assert out["subsets"]["low"]["n_boards"] == 2
    assert out["subsets"]["low"]["board_accuracy"] == 0.5
    assert out["subsets"]["high"]["n_boards"] == 1
    assert out["subsets"]["high"]["board_accuracy"] == 1.0


def test_aggregate_subsets_length_mismatch():
    gt = _start()
    with pytest.raises(ValueError):
        aggregate_subsets([gt, gt], [gt, gt], ["low"])
