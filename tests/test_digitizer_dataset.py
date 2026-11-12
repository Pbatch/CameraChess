"""Digitizer: parsing moves.txt, replaying it into one placement per frame, cropping and loading (no network)."""

import json

import pytest

from camera_chess.chessqueries.core import Split
from camera_chess.chessqueries.data import DatasetName, Digitizer
from camera_chess.chessqueries.data.base import DATASET_REGISTRY
from camera_chess.chessqueries.data.digitizer import (
    GAMES,
    TEST_GAMES,
    VAL_GAMES,
    crop_box,
    game_split,
    parse_moves,
    placements,
)

MOVES = """2
https://www.chess.com/game/live/106586880451?username=oscar1618
432 8 1504 21 440 1057 1476 1066
1. e4 d6
2. Nf3 g6
3. Qxf7# 1-0
"""


def test_registered():
    assert DATASET_REGISTRY[DatasetName.DIGITIZER] is Digitizer


def test_parses_rotation_corners_and_moves():
    rotation, corners, moves = parse_moves(MOVES)
    assert rotation == 2
    assert corners == [(432, 8), (1504, 21), (440, 1057), (1476, 1066)]
    assert moves == ["e4", "d6", "Nf3", "g6", "Qxf7#"]


def test_one_placement_per_frame():
    fens = placements(["e4", "d6", "Nf3", "g6"])
    assert len(fens) == 5
    assert fens[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert fens[-1] == "rnbqkbnr/ppp1pp1p/3p2p1/8/4P3/5N2/PPPP1PPP/RNBQKB1R"


def test_illegal_move_fails():
    with pytest.raises(ValueError):
        placements(["e5"])


def test_crop_is_the_corners_box_within_the_frame():
    corners = [(100, 10), (1100, 10), (100, 1110), (1100, 1110)]
    assert crop_box(corners, 1920, 1080) == (100, 10, 1100, 1080)


def test_splits_are_whole_games_and_disjoint():
    assert not TEST_GAMES & VAL_GAMES
    assert {15, 20} <= TEST_GAMES  # the same game recorded twice
    assert {game_split(g) for g in GAMES} == {Split.TRAIN, Split.VAL, Split.TEST}


def test_loads_a_split(tmp_path, monkeypatch):
    monkeypatch.setattr(Digitizer, "expected_samples", {Split.TRAIN: 1, Split.VAL: 0, Split.TEST: 1})
    annotations = []
    for game, split in [(1, "train"), (6, "test")]:
        image = f"images/{split}/digitizer_g{game:02d}_000.jpg"
        (tmp_path / image).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / image).write_bytes(b"crop")
        annotations.append(
            {
                "image": image,
                "fen": "8/8/8/8/8/8/8/K6k",
                "split": split,
                "game": game,
                "ply": 0,
                "corners": [[0, 0], [1, 0], [0, 1], [1, 1]],
                "box": [0, 0, 2, 2],
            }
        )
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    test = Digitizer(tmp_path).torch_dataset(Split.TEST).samples
    assert [s.sample_id for s in test] == ["digitizer_g06_000"]
    assert test[0].meta["game"] == 6
