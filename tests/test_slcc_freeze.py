"""Freeze pins for the released SLCC manifest (``data/slcc/dataset/annotations.json``).

The manifest lives under the gitignored ``data/`` tree, so these tests **skip**
when it isn't present (e.g. CI without the dataset) and act as a drift guard for
anyone who does have it. Bumping the dataset is a deliberate act: regenerate the
manifest, confirm the new numbers, then update the constants below in the same
commit.
"""

import collections
import hashlib
import json

import chess
import pytest

from camera_chess.chessqueries.core import NUM_PIECES, Board
from camera_chess.chessqueries.data.slcc import SLCC

# --- Frozen release identity (SLCC v1) -------------------------------------
# The manifest is reconstructed from the release, so its bytes can change (serializer, fingerprints) while the
# data doesn't. Pin the upstream release manifest it was built from, and the fields that define the benchmark.
RELEASE_MANIFEST_SHA256 = "3c6137508de17472bd97470cd3f7d218e585db9f24a77574165029d16401e375"
CONTENT_FIELDS = ("sample_id", "split", "gt_fen", "game_id", "video_id", "frame_index", "crop_bbox")
CONTENT_SHA256 = "0cea0e9f9ebc85d4d3779cfe23ecec84e996e84d0f8a16dc8da9034d499a0176"
N_SAMPLES = 2174
SPLIT_COUNTS = {"train": 1475, "val": 326, "test": 373}
N_GAMES = 152
N_VIDEOS = 20


@pytest.fixture(scope="module")
def manifest():
    path = SLCC().manifest_path
    if not path.is_file():
        # ty misreads pytest's _WithException wrapper's signature
        pytest.skip(f"SLCC manifest not present at {path} (data/ is local-only)")  # ty: ignore[too-many-positional-arguments]
    return path


def test_manifest_is_built_from_frozen_release(manifest):
    """The upstream release manifest the reconstruction was built from — the release identifier."""
    got = json.loads(manifest.read_text())["provenance"]["release_manifest_sha256"]
    assert got == RELEASE_MANIFEST_SHA256, (
        f"SLCC manifest was built from a different release ({got}). If this is an intentional "
        f"dataset bump, update the pins and the counts in this file."
    )


def test_manifest_content_is_frozen(manifest):
    """Samples, splits, labels and crops, independent of how the file is serialized."""
    samples = json.loads(manifest.read_text())["samples"]
    rows = sorted([sample[field] for field in CONTENT_FIELDS] for sample in samples)
    got = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    assert got == CONTENT_SHA256, (
        f"SLCC manifest content changed ({got}). If this is an intentional "
        f"dataset bump, update the pins and the counts in this file."
    )


def test_sample_and_split_counts(manifest):
    samples = json.loads(manifest.read_text())["samples"]
    assert len(samples) == N_SAMPLES
    counts = collections.Counter(s["split"] for s in samples)
    assert dict(counts) == SPLIT_COUNTS
    assert len({s["video_id"] for s in samples}) == N_VIDEOS
    assert len({s["game_id"] for s in samples}) == N_GAMES


def test_splits_are_group_disjoint_by_game(manifest):
    """No game may straddle splits — the leakage guarantee for the benchmark."""
    samples = json.loads(manifest.read_text())["samples"]
    by_game = collections.defaultdict(set)
    for s in samples:
        by_game[s["game_id"]].add(s["split"])
    straddling = {g for g, sp in by_game.items() if len(sp) > 1}
    assert not straddling, f"{len(straddling)} game(s) leak across splits: {straddling}"


def test_every_fen_is_legal_and_human_verified(manifest):
    samples = json.loads(manifest.read_text())["samples"]
    for s in samples:
        chess.Board(s["gt_fen"])  # full FEN is legal
        labels = Board.from_fen(s["gt_fen"]).labels
        assert len(labels) == 64 and all(0 <= c < NUM_PIECES for c in labels)
        assert s["verified_by_human"] is True
