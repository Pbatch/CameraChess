"""SLCC broadcast dataset: human-reviewed frames reconstructed from relay videos.

The manifest (``data/slcc/dataset/annotations.json``) holds one record per frame,
each carrying its own ``split``. It is built by the upstream ChessQueries tooling,
which assigns splits by game so frames of one game never straddle train and test.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from types import MappingProxyType

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config

MANIFEST_NAME = "annotations.json"

# Sample fields copied straight onto BoardSample.meta (provenance for review/eval).
_META_KEYS = (
    "video_id",
    "frame_index",
    "timestamp_s",
    "game_id",
    "round_id",
    "ply",
    "side_to_move",
    "players",
    "template_id",
    "crop_bbox",
    "confidence",
    "source",
    "requires_review",
    "verified_by_human",
)


class SLCC(ChessDataset):
    name = DatasetName.SLCC
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 1_475, Split.VAL: 326, Split.TEST: 373})

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root)
        self.manifest_path = self.root / MANIFEST_NAME

    @staticmethod
    def default_root() -> Path:
        return get_config().DATA_ROOT / "slcc" / "dataset"

    def _manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"SLCC manifest not found at {self.manifest_path}. "
                "Reconstruct SLCC with the upstream ChessQueries tooling (see DEVELOPMENT.md)."
            )
        return json.loads(self.manifest_path.read_text())

    @staticmethod
    def _sample_id(record: dict) -> str:
        return record.get("sample_id") or Path(record["image"]).stem

    def _structural_issues(self) -> tuple[str, ...]:
        """Problems with the whole reconstruction, which a single complete split would hide."""
        manifest = self._manifest()
        records = manifest["samples"]
        issues: list[str] = []
        if manifest.get("provenance", {}).get("partial") is True:
            issues.append("manifest provenance marks this reconstruction partial")

        for label, values in (
            ("sample IDs", [self._sample_id(r) for r in records]),
            ("image paths", [r["image"] for r in records]),
        ):
            duplicates = sorted(v for v, n in Counter(values).items() if n > 1)
            if duplicates:
                issues.append(f"duplicate {label}: {duplicates[:5]}")

        game_splits: dict[str, set[str]] = defaultdict(set)
        for r in records:
            game_splits[r["game_id"]].add(r["split"])
        straddling = sorted(game for game, splits in game_splits.items() if len(splits) > 1)
        if straddling:
            issues.append(f"game IDs cross splits: {straddling[:5]}")

        actual = Counter(r["split"] for r in records)
        expected = {split.value: count for split, count in self.expected_samples.items() if split is not None}
        if dict(actual) != expected:
            issues.append(f"full split counts are {dict(actual)}, expected {expected}")
        missing = sum(not (self.root / r["image"]).is_file() for r in records)
        if missing:
            issues.append(f"{missing} crop image(s) are missing across the full reconstruction")
        return tuple(issues)

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        return [
            BoardSample(
                image_path=self.root / r["image"],
                board=Board.from_fen(r["gt_fen"]),
                dataset=self.name,
                sample_id=self._sample_id(r),
                split=Split(r["split"]),
                meta={"gt_fen": r["gt_fen"], **{k: r[k] for k in _META_KEYS if k in r}},
            )
            for r in self._manifest()["samples"]
            if split is None or r["split"] == split.value
        ]
