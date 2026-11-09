import json

import numpy as np

from camera_chess.labels.frames import CLASSES, Frame, rotate
from camera_chess.labels.make_chessqueries_dataset import (
    choose_rotation,
    load_label_fixes,
    orientation_penalty,
    output_split,
)


def frame(squares, classes, split="train", group="train/game"):
    return Frame(
        name="game_1",
        split=split,
        image=None,  # ty: ignore[invalid-argument-type]
        group=group,
        squares=np.array(squares),
        classes=np.array([CLASSES.index(c) for c in classes]),
        off_board=0,
        H=np.eye(3),
    )


# White on ranks 1-2 and Black on ranks 7-8, as in a game's opening
OPENING = frame(
    [[4, 0], [4, 1], [3, 1], [4, 7], [4, 6], [3, 6]],
    ["white-king", "white-pawn", "white-pawn", "black-king", "black-pawn", "black-pawn"],
)


def test_upright_board_has_the_lowest_penalty():
    penalties = [orientation_penalty([OPENING], k) for k in range(4)]
    assert int(np.argmin(penalties)) == 0
    # Upside down, White sits above Black
    assert penalties[2] > penalties[0]


def test_choose_rotation_undoes_a_rotated_grid():
    # The lattice came out rotated by 1, so the rotation putting a1 bottom-left is 3
    turned = frame(rotate(OPENING.squares, 1), [CLASSES[c] for c in OPENING.classes])
    k, margin = choose_rotation([turned])
    assert k == 3 and margin > 0


def test_pawns_on_a_back_rank_are_penalised():
    # On d1, or on d8 turned upside down; a quarter turn puts it on a middle rank
    pawn = frame([[3, 0]], ["white-pawn"])
    assert orientation_penalty([pawn], 0) == 10
    assert orientation_penalty([pawn], 2) == 10
    assert orientation_penalty([pawn], 1) == 0


def test_output_split_holds_out_whole_videos():
    assert output_split(frame([], [], split="val", group="val/game"), test_fraction=1.0) == "val"
    # Photos are their own group, without a split prefix, and stay in train
    assert output_split(frame([], [], group="roboflow_12_3"), test_fraction=1.0) == "train"
    assert output_split(frame([], [], group="train/game"), test_fraction=1.0) == "test"
    assert output_split(frame([], [], group="train/game"), test_fraction=0.0) == "train"
    groups = [f"train/video{i}" for i in range(400)]
    test = [g for g in groups if output_split(frame([], [], group=g), test_fraction=0.12) == "test"]
    assert 20 < len(test) < 80
    assert test == [g for g in groups if output_split(frame([], [], group=g), test_fraction=0.12) == "test"]


def test_later_reviews_override_earlier_ones(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "game_1": {"verdict": "model", "square": "e2", "label": "P", "model": "p"},
                "game_2": {"verdict": "model", "square": "a1", "label": "R", "model": ""},
                "game_3": {"verdict": "unsure", "square": "a1", "label": "R", "model": ""},
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "game_2:a1": {"verdict": "label", "square": "a1", "label": "R", "model": ""},
                "game_2:h1": {"verdict": "model", "square": "h1", "label": "", "model": "R"},
            }
        )
    )
    fixes = load_label_fixes([first, second, tmp_path / "missing.json"])
    assert fixes == {"game_1": {"e2": ["P", "p"]}, "game_2": {"h1": ["", "R"]}}
