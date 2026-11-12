"""Samryan: reading placements from file names, finding the board by matching its warp, cropping and loading
(no network)."""

import json

import numpy as np
import pytest
from PIL import Image

from camera_chess.chessqueries.core import Split
from camera_chess.chessqueries.data import DatasetName, Samryan
from camera_chess.chessqueries.data.base import DATASET_REGISTRY
from camera_chess.chessqueries.data.samryan import (
    board_corners,
    crop_box,
    git_blob_sha,
    placement_from_name,
    valid_corners,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_registered():
    assert DATASET_REGISTRY[DatasetName.SAMRYAN] is Samryan


@pytest.mark.parametrize(
    ("name", "placement"),
    [
        ("rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.JPG", START),
        ("rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR_3.JPG", START),
        ("1B2b3-Kp6-8-8-2k5-8-8-8.JPG", "1B2b3/Kp6/8/8/2k5/8/8/8"),
        ("rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP.JPG", None),  # seven ranks
        ("rnbqkbnr-ppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.JPG", None),  # a seven-square rank
        ("IMG_0001.JPG", None),
    ],
)
def test_placement_from_name(name, placement):
    assert placement_from_name(name) == placement


def test_git_blob_sha():
    # `git hash-object` of a file holding "hello\n"
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_rejects_a_twisted_or_lopsided_board():
    square = np.array([[100, 100], [900, 100], [100, 900], [900, 900]], dtype=np.float32)
    assert valid_corners(square, 1000, 1000)
    assert not valid_corners(square[[1, 0, 2, 3]], 1000, 1000)  # TL and TR swapped
    assert not valid_corners(np.array([[100, 100], [900, 100], [100, 500], [900, 900]], dtype=np.float32), 1000, 1000)
    assert not valid_corners(square, 800, 1000)  # past the photo's right edge


def _board(size: int) -> Image.Image:
    """A textured chessboard: SIFT needs more than flat squares to match"""
    rng = np.random.default_rng(0)
    squares = np.kron((np.indices((8, 8)).sum(0) % 2) * 160 + 40, np.ones((size // 8, size // 8)))
    noise = rng.integers(0, 60, (size // 16, size // 16)).repeat(16, 0).repeat(16, 1)
    return Image.fromarray((squares + noise).astype(np.uint8)).convert("RGB")


def test_finds_the_board_from_its_warp(tmp_path):
    board = _board(640)
    photo = Image.new("RGB", (1600, 1200), (90, 60, 30))
    photo.paste(board, (500, 300))
    photo.save(tmp_path / "photo.jpg", quality=95)
    board.resize((700, 700)).save(tmp_path / "warp.png")
    corners = board_corners(tmp_path / "photo.jpg", tmp_path / "warp.png")
    assert corners is not None
    np.testing.assert_allclose(corners, [(500, 300), (1140, 300), (500, 940), (1140, 940)], atol=4)


def test_no_board_when_the_warp_is_of_something_else(tmp_path):
    rng = np.random.default_rng(1)
    Image.new("RGB", (1600, 1200), (90, 60, 30)).save(tmp_path / "photo.jpg")
    Image.fromarray(rng.integers(0, 255, (700, 700, 3), dtype=np.uint8)).save(tmp_path / "warp.png")
    assert board_corners(tmp_path / "photo.jpg", tmp_path / "warp.png") is None


def test_crop_is_the_corners_box_within_the_photo():
    corners = [(100.0, 10.0), (1100.0, 10.0), (100.0, 1110.0), (1100.0, 1110.0)]
    assert crop_box(corners, 1920, 1080) == (100, 10, 1100, 1080)


def test_loads_a_split(tmp_path, monkeypatch):
    monkeypatch.setattr(Samryan, "expected_samples", {Split.TRAIN: 1, Split.VAL: 0, Split.TEST: 1})
    annotations = []
    for n, split in enumerate(["train", "test"]):
        image = f"images/{split}/samryan_{n:03d}.jpg"
        (tmp_path / image).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / image).write_bytes(b"crop")
        annotations.append(
            {
                "image": image,
                "fen": START,
                "split": split,
                "group": f"samryan_g{n:03d}",
                "source": f"{n}.JPG",
                "corners": [[0, 0], [1, 0], [0, 1], [1, 1]],
                "box": [0, 0, 2, 2],
                "scale": 1.0,
            }
        )
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    test = Samryan(tmp_path).torch_dataset(Split.TEST).samples
    assert [s.sample_id for s in test] == ["samryan_001"]
    assert test[0].meta["group"] == "samryan_g001"
