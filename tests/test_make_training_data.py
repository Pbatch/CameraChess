import json

import chess
import numpy as np
import pytest

from camera_chess.videos import make_training_data
from camera_chess.videos.make_training_data import (
    AFTER,
    BEFORE,
    CLASSES,
    LEAD_OUT,
    OFFSET,
    PER_POSITION,
    crop,
    crop_box,
    frame_times,
    labels,
    positions,
)

GAME = {"moves": ["e2e4", "e7e5", "g1f3"], "times": [10.0, 20.0, 21.0]}


def test_labels_are_in_fen_order():
    out = labels(chess.Board())
    assert "".join(CLASSES[i] for i in out[:8]) == "rnbqkbnr"
    assert "".join(CLASSES[i] for i in out[-8:]) == "RNBQKBNR"
    assert (out[16:48] == 0).all()


def test_positions_run_from_move_to_move():
    out = positions(GAME)
    assert [board.board_fen() for board, _, _ in out][-1] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    assert [(start, end) for _, start, end in out] == [
        pytest.approx((10 + OFFSET, 20 + OFFSET)),
        pytest.approx((20 + OFFSET, 21 + OFFSET)),
        pytest.approx((21 + OFFSET, 21 + OFFSET + LEAD_OUT)),
    ]


def test_frame_times_keep_clear_of_moves():
    chosen = frame_times(GAME)
    by_ply = {}
    for board, t in chosen:
        by_ply.setdefault(board.ply(), []).append(t)
    # After e7e5 the position stands 1s, too short once the windows around moves are cut
    assert set(by_ply) == {1, 3}
    assert len(by_ply[1]) == PER_POSITION
    assert all(10 + OFFSET + AFTER < t < 20 + OFFSET - BEFORE for t in by_ply[1])
    # The last position has no move after it, so runs to LEAD_OUT
    assert all(21 + OFFSET + AFTER < t < 21 + OFFSET + LEAD_OUT for t in by_ply[3])


def test_frame_times_are_capped_per_video(monkeypatch):
    monkeypatch.setattr(make_training_data, "MAX_FRAMES", 4)
    board, moves = chess.Board(), []
    for _ in range(10):
        move = next(iter(board.legal_moves))
        board.push(move)
        moves.append(move.uci())
    chosen = frame_times({"moves": moves, "times": [10.0 * (i + 1) for i in range(10)]})
    times = [t for _, t in chosen]
    assert len(times) == 4 and times == sorted(times)
    assert chosen[0][0].ply() == 1 and chosen[-1][0].ply() == 10


def test_crop_box_is_the_board_region_within_the_frame():
    # Board region 100x100 at (50, 120)
    assert crop_box((50, 120, 100, 100), width=500, height=500) == (50, 120, 150, 220)
    assert crop_box((50, 150, 100, 100), width=120, height=230) == (50, 150, 120, 230)


def test_crop_blacks_out_the_overlay():
    frame = np.full((100, 100, 3), 255, np.uint8)
    out = crop(frame, (10, 10, 60, 60), overlay=(50, 0, 40, 30))
    assert out.shape == (50, 50, 3)
    assert (out[:20, 40:] == 0).all()
    assert (out[20:] == 255).all() and (out[:, :40] == 255).all()


def test_benchmark_games_are_refused(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark.txt"
    benchmark.write_text("D81hFXZ0Cp4\n")
    monkeypatch.setattr(make_training_data, "benchmark_path", lambda: benchmark)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="tracker replay"):
        make_training_data.main(["yunker/other", "yunker/D81hFXZ0Cp4_game_2"], out)
    # Refused before the output folder is rebuilt
    assert out.exists()


@pytest.fixture
def dataset(monkeypatch):
    """A dataset whose probabilities read the start position after e4 on every frame"""
    board = chess.Board()
    board.push_uci("e2e4")
    probs = np.zeros((400, 64, len(CLASSES)), np.float32)
    probs[:, np.arange(64), labels(board)] = 1
    monkeypatch.setattr(make_training_data, "load_config", lambda d: {"roi": [0, 0, 8, 8], "overlay_roi": [0, 0, 1, 1]})
    monkeypatch.setattr(make_training_data, "load_probs", lambda d: (probs, {"fps": 4, "roi": [0, 0, 8, 8]}))
    return probs


def test_videos_the_model_mostly_disagrees_with_are_skipped(dataset, monkeypatch, tmp_path):
    # Every position after the first is a different one, so most frames disagree
    game = {"moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"], "times": [10.0, 20.0, 30.0, 40.0, 50.0]}
    monkeypatch.setattr(make_training_data, "load_game", lambda d: game)
    dataset[:] = 0
    dataset[:, :, 0] = 1
    assert make_training_data.make("yunker/x", tmp_path) == []


def test_probabilities_for_another_region_are_refused(dataset, monkeypatch, tmp_path):
    monkeypatch.setattr(make_training_data, "load_game", lambda d: GAME)
    monkeypatch.setattr(make_training_data, "load_config", lambda d: {"roi": [1, 1, 8, 8], "overlay_roi": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="another region"):
        make_training_data.make("yunker/x", tmp_path)


# White's pawn takes the rook on g8 and promotes; Black's knight then takes the new piece
PROMOTION = {
    "moves": ["e2e4", "f7f5", "e4f5", "g7g6", "f5g6", "g8f6", "g6h7", "h8g8", "h7g8q", "f6g8"],
    "times": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 40.0],
}


def piece_on(board, square):
    piece = board.piece_at(chess.parse_square(square))
    return None if piece is None else piece.symbol()


def test_a_pawn_left_on_the_board_is_labelled_as_the_pawn_until_its_square_changes():
    out = positions({**PROMOTION, "pawns_left": {8: None}})
    assert len(out) == len(PROMOTION["moves"])
    # The promotion is the true move, but a white pawn stood on g8 until the knight took it
    board, start, end = out[8]
    assert piece_on(board, "g8") == "P"
    assert (start, end) == (pytest.approx(26 + OFFSET), pytest.approx(40 + OFFSET))
    assert piece_on(out[9][0], "g8") == "n"


def test_a_pawn_swapped_later_splits_its_position_at_the_swap():
    out = positions({**PROMOTION, "pawns_left": {8: 30.0}})
    assert len(out) == len(PROMOTION["moves"]) + 1
    assert [(piece_on(board, "g8"), start, end) for board, start, end in out[8:10]] == [
        ("P", pytest.approx(26 + OFFSET), 30.0),
        ("Q", 30.0, pytest.approx(40 + OFFSET)),
    ]
    # Frames keep clear of the hand making the swap, as of a move
    chosen = [(piece_on(board, "g8"), t) for board, t in frame_times({**PROMOTION, "pawns_left": {8: 30.0}})]
    assert all(t < 30.0 - BEFORE for symbol, t in chosen if symbol == "P")
    assert all(30.0 + AFTER < t < 40 + OFFSET - BEFORE for symbol, t in chosen if symbol == "Q")


def test_add_keeps_the_other_datasets_and_redoes_these(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark.txt"
    benchmark.write_text("")
    monkeypatch.setattr(make_training_data, "benchmark_path", lambda: benchmark)
    out = tmp_path / "out"
    (out / "images" / "train").mkdir(parents=True)
    old = [{"image": f"images/train/{g}.jpg", "group": f"yunker/{g}"} for g in ("kept", "redone")]
    for a in old:
        (out / a["image"]).write_bytes(b"")
    (out / "annotations.json").write_text(json.dumps(old))

    def make(dataset, out_dir, checked=True):
        image = f"images/train/{dataset.split('/')[1]}_new.jpg"
        (out_dir / image).write_bytes(b"")
        return [{"image": image, "group": dataset}]

    monkeypatch.setattr(make_training_data, "make", make)
    make_training_data.main(["yunker/redone", "yunker/added"], out, add=True)
    images = [a["image"] for a in json.loads((out / "annotations.json").read_text())]
    assert images == ["images/train/kept.jpg", "images/train/redone_new.jpg", "images/train/added_new.jpg"]
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == [
        "added_new.jpg",
        "kept.jpg",
        "redone_new.jpg",
    ]
