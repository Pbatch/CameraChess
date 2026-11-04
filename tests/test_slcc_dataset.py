"""SLCC loader (no pixels/network)."""

import json

import pytest

from camera_chess.chessqueries.core import Split
from camera_chess.chessqueries.data import DatasetName, get_dataset
from camera_chess.chessqueries.data.base import DATASET_REGISTRY, DatasetIncompleteError
from camera_chess.chessqueries.data.slcc import MANIFEST_NAME, SLCC

FEN = "3b4/3B2k1/5pb1/p1pN3p/P1P1P2P/1P3K2/8/8 b - - 2 65"


def _rec(sid: str, split: str, video_id: str = "vidA", game_id: str = "g") -> dict:
    return {
        "image": f"images/{sid}.jpg",
        "gt_fen": FEN,
        "split": split,
        "video_id": video_id,
        "game_id": game_id,
        "ply": 1,
        "side_to_move": "b",
        "players": ["W", "B"],
        "template_id": "t",
        "confidence": 0.9,
        "verified_by_human": True,
    }


def _write(root, recs):
    (root).mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(json.dumps({"version": "v1", "samples": recs}))
    for record in recs:
        image = root / record["image"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"synthetic image")


def test_registered():
    assert DATASET_REGISTRY[DatasetName.SLCC] is SLCC
    assert get_dataset(DatasetName.SLCC).splits == (Split.TRAIN, Split.VAL, Split.TEST)


def test_loader_filters_by_split(tmp_path):
    root = tmp_path / "dataset"
    _write(root, [_rec("a", "train"), _rec("b", "train"), _rec("c", "val")])
    ds = SLCC(root)
    assert len(ds.torch_dataset(Split.TRAIN, allow_partial=True).samples) == 2
    val = ds.torch_dataset(Split.VAL, allow_partial=True).samples
    assert len(val) == 1
    s = val[0]
    assert s.dataset is DatasetName.SLCC and s.split is Split.VAL
    assert s.sample_id == "c"
    assert s.board.placement == FEN.split()[0]  # placement round-trips through Board
    assert s.meta["verified_by_human"] is True


def test_complete_requested_split_does_not_hide_partial_full_reconstruction(tmp_path):
    root = tmp_path / "dataset"
    _write(root, [_rec(f"test-{index}", "test") for index in range(373)])
    dataset = SLCC(root)

    with pytest.raises(DatasetIncompleteError, match="full split counts"):
        dataset.torch_dataset(Split.TEST)

    loaded = dataset.torch_dataset(Split.TEST, allow_partial=True)
    assert len(loaded.samples) == 373
    assert loaded.completeness is not None and loaded.completeness.structural_issues


def test_manifest_rejects_a_game_that_crosses_splits(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    _write(
        root,
        [
            _rec("train", "train", game_id="shared"),
            _rec("val", "val", game_id="val-game"),
            _rec("test", "test", game_id="shared"),
        ],
    )
    monkeypatch.setattr(
        SLCC,
        "expected_samples",
        {Split.TRAIN: 1, Split.VAL: 1, Split.TEST: 1},
    )

    with pytest.raises(DatasetIncompleteError, match="game IDs cross splits"):
        SLCC(root).torch_dataset(Split.TEST)


def test_manifest_reports_duplicate_sample_and_image_identities(tmp_path):
    root = tmp_path / "dataset"
    _write(
        root,
        [
            _rec("same", "train", game_id="train-game"),
            _rec("same", "val", game_id="val-game"),
        ],
    )

    issues = SLCC(root)._structural_issues()

    assert any("duplicate sample IDs" in issue for issue in issues)
    assert any("duplicate image paths" in issue for issue in issues)
