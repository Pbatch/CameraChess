"""Yunker crop loader (no pixels), and joint training with a source that has no VAL split."""

import json
from types import SimpleNamespace

import pytest

from camera_chess.chessqueries.core import Split
from camera_chess.chessqueries.data import DatasetName, Yunker
from camera_chess.chessqueries.data.base import DATASET_REGISTRY
from camera_chess.chessqueries.train import lit

FEN = "3b4/3B2k1/5pb1/p1pN3p/P1P1P2P/1P3K2/8/8"


def _crops(root, names: list[str]) -> None:
    (root / "images" / "train").mkdir(parents=True)
    annotations = []
    for name in names:
        image = f"images/train/{name}.jpg"
        (root / image).write_bytes(b"crop")
        annotations.append(
            {"image": image, "fen": FEN, "split": "train", "group": "yunker/abc", "box": [10, 20, 300, 200]}
        )
    (root / "annotations.json").write_text(json.dumps(annotations))


def test_registered():
    assert DATASET_REGISTRY[DatasetName.YUNKER] is Yunker


def test_loads_train_crops_with_their_board_box(tmp_path):
    _crops(tmp_path, ["yunker_abc_001000", "yunker_abc_002000"])
    ds = Yunker(tmp_path)
    assert ds.splits == (Split.TRAIN,)
    train = ds.torch_dataset(Split.TRAIN).samples
    assert [s.sample_id for s in train] == ["yunker_abc_001000", "yunker_abc_002000"]
    assert train[0].board.placement == FEN
    assert train[0].meta["box"] == [10, 20, 300, 200]


def _fake_source(splits):
    return SimpleNamespace(splits=splits, torch_dataset=lambda split, **_: [split])


def test_joint_training_skips_val_for_a_source_without_one(monkeypatch):
    sources = {DatasetName.CHESSCAM: (Split.TRAIN, Split.VAL), DatasetName.YUNKER: (Split.TRAIN,)}
    monkeypatch.setattr(lit, "get_dataset", lambda name: _fake_source(sources[name]))
    module = lit.JointDataModule(tuple(sources), resolution=448, batch_size=2, workers=0)
    module.setup()
    assert len(module.train_ds) == 2
    assert module.val_ds is not None and len(module.val_ds) == 1


def test_joint_training_on_all_splits_has_no_val(monkeypatch):
    sources = {DatasetName.CHESSCAM: (Split.TRAIN, Split.VAL, Split.TEST), DatasetName.YUNKER: (Split.TRAIN,)}
    monkeypatch.setattr(lit, "get_dataset", lambda name: _fake_source(sources[name]))
    module = lit.JointDataModule(tuple(sources), resolution=448, batch_size=2, workers=0, all_splits=True)
    module.setup()
    assert len(module.train_ds) == 4
    assert module.val_ds is None
    assert module.val_dataloader() == []


def test_joint_training_needs_some_val_split(monkeypatch):
    monkeypatch.setattr(lit, "get_dataset", lambda name: _fake_source((Split.TRAIN,)))
    module = lit.JointDataModule((DatasetName.YUNKER,), resolution=448, batch_size=2, workers=0)
    with pytest.raises(ValueError, match="VAL"):
        module.setup()


def test_training_imports_only_the_loaders_it_uses():
    """The registry loads a dataset's module on first use, so training doesn't need cv2 or python-chess"""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import camera_chess.chessqueries.train.training_job\n"
        "from camera_chess.chessqueries.data import DatasetName, get_dataset\n"
        "from camera_chess.chessqueries.data.base import DATASET_REGISTRY\n"
        "assert DatasetName.YUNKER not in DATASET_REGISTRY\n"
        "try:\n"
        "    get_dataset(DatasetName.YUNKER)\n"
        "except Exception:\n"
        "    pass\n"
        "assert DatasetName.YUNKER in DATASET_REGISTRY\n"
        "loaded = {m for m in ('cv2', 'chess') if m in sys.modules}\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
