"""Fenit: reading placements from file names, board corners from labels, grouping games, splitting and loading
(no network)."""

import json

import pytest

from camera_chess.chessqueries.core import Split
from camera_chess.chessqueries.data import DatasetName, Fenit
from camera_chess.chessqueries.data.base import DATASET_REGISTRY
from camera_chess.chessqueries.data.fenit import (
    MAX_HELD_OUT_GAME,
    assign_splits,
    board_corners,
    crop_box,
    games,
    placement_from_name,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def test_registered():
    assert DATASET_REGISTRY[DatasetName.FENIT] is Fenit


@pytest.mark.parametrize(
    ("name", "placement"),
    [
        ("1k1r3r_7p_p1p1Rp2_1pbp3b_5P1P_6PB_PPP5_RK6.jpeg", "1k1r3r/7p/p1p1Rp2/1pbp3b/5P1P/6PB/PPP5/RK6"),
        ("rnbqkbnr-pppppppp-8-8-4P3-8-PPPP1PPP-RNBQKBNR.jpg", AFTER_E4),
        ("8_8_1p1k4_5ppp_PPK1p3_6P1_5PP1_8 b (2).jpg", "8/8/1p1k4/5ppp/PPK1p3/6P1/5PP1/8"),
        ("rnbqkbnr_pppppppp_8_8_8_8_PPPPPPPP_RNBQKBNR (full board).jpeg", START),
        ("8_8_8_8_8_8_8_8(board).jpeg", "8/8/8/8/8/8/8/8"),
    ],
)
def test_reads_the_placement_from_the_name(name, placement):
    assert placement_from_name(name) == placement


@pytest.mark.parametrize("name", ["20260219_222546.jpg", "K.jpg", "b (2).jpg", "R7_5pk1_5p1_P6p_7P_r5P1_5P2_5K2 b.jpg"])
def test_names_without_a_valid_placement(name):
    assert placement_from_name(name) is None


def test_board_corners_come_from_the_board_row_only():
    label = (
        "8 0.5 0.5 0.2 0.3 0.4 0.5 2 0.5 0.6 2 0.6 0.5 2 0.5 0.3 2\n"
        "0 0.5 0.5 0.9 0.5 0.03 0.38 2 0.43 0.75 2 0.96 0.42 2 0.51 0.19 2\n"
    )
    assert board_corners(label) == [(0.03, 0.38), (0.43, 0.75), (0.96, 0.42), (0.51, 0.19)]
    assert board_corners(label.splitlines()[0]) is None


def test_crop_is_the_corners_box_in_pixels():
    corners = [(0.1, 0.0), (0.6, 0.0), (0.1, 0.5), (0.6, 0.5)]
    assert crop_box(corners, 1000, 2000) == (100, 0, 600, 1000)


def test_games_link_close_positions_taken_with_one_camera():
    photos = [
        ("a.jpg", START, (3000, 4000)),
        ("b.jpg", AFTER_E4, (3000, 4000)),  # two squares from a
        ("c.jpg", AFTER_E4, (4284, 5712)),  # another camera
        ("d.jpg", "8/8/8/8/8/8/8/K6k", (3000, 4000)),  # another game
    ]
    assert sorted(sorted(g) for g in games(photos)) == [[0, 1], [2], [3]]


def test_splits_keep_games_whole_and_big_games_in_train():
    big = list(range(MAX_HELD_OUT_GAME + 1))
    small = [[n] for n in range(len(big), len(big) + 20)]
    splits = assign_splits([big, *small])
    assert {splits[i] for i in big} == {Split.TRAIN}
    assert set(splits.values()) == {Split.TRAIN, Split.VAL, Split.TEST}


def test_loads_a_split(tmp_path, monkeypatch):
    monkeypatch.setattr(Fenit, "expected_samples", {Split.TRAIN: 1, Split.VAL: 0, Split.TEST: 1})
    annotations = []
    for n, split in [(0, "train"), (1, "test")]:
        image = f"images/{split}/fenit_{n:03d}.jpg"
        (tmp_path / image).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / image).write_bytes(b"crop")
        annotations.append(
            {
                "image": image,
                "fen": START,
                "split": split,
                "group": f"fenit_g{n:03d}",
                "source": f"{n}.jpg",
                "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "box": [0, 0, 2, 2],
                "scale": 1.0,
            }
        )
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    test = Fenit(tmp_path).torch_dataset(Split.TEST).samples
    assert [s.sample_id for s in test] == ["fenit_001"]
    assert test[0].meta["group"] == "fenit_g001"
    assert test[0].board.placement == START
