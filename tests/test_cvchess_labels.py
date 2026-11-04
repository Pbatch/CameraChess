"""Validate the vendored CVChess labels file (no images required)."""

import itertools

import chess

from camera_chess.chessqueries.core import NUM_PIECES, Board
from camera_chess.chessqueries.data.cvchess import load_labels


def test_labels_parse_and_convert():
    labels = load_labels()
    assert len(labels) == 352
    for e in labels:
        assert e["image"].endswith(".jpg")
        # full FEN is legal
        chess.Board(e["gt_fen"])
        # placement converts to exactly 64 class ids in our vocab
        lab = Board.from_fen(e["gt_fen"]).labels
        assert len(lab) == 64
        assert all(0 <= c < NUM_PIECES for c in lab)


def test_first_frame_is_start_position():
    """Content pin: the first frame (by image order) must show the START position.

    Guards against the upstream off-by-one (FENgen ``include_start=False``), where
    every frame was labelled one ply ahead of what it shows. A globally-shifted
    sequence is still a *coherent game*, so the coherence check below cannot catch
    it — only anchoring to image content can.
    """
    import re

    START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    labels = sorted(load_labels(), key=lambda e: int(re.findall(r"\d+", e["image"])[0]))
    assert labels[0]["gt_fen"].split()[0] == START, (
        f"first frame {labels[0]['image']} is not the start position "
        f"(got {labels[0]['gt_fen'].split()[0]!r}) — labels likely off by one ply"
    )


def test_progression_is_coherent():
    """Consecutive frames are the same position or one legal move apart."""
    import re

    labels = sorted(load_labels(), key=lambda e: int(re.findall(r"\d+", e["image"])[0]))
    jumps = 0
    for a, b in itertools.pairwise(labels):
        fa, fb = a["gt_fen"].split(" ")[0], b["gt_fen"].split(" ")[0]
        if fa == fb:
            continue
        board = chess.Board(a["gt_fen"])
        reachable = False
        for mv in board.legal_moves:
            board.push(mv)
            if board.board_fen() == fb:
                reachable = True
            board.pop()
            if reachable:
                break
        if not reachable:
            jumps += 1
    # A handful of game boundaries could in principle jump; here we expect none.
    assert jumps == 0, f"{jumps} incoherent frame transitions"


def test_splits_keep_runs_of_positions_together():
    """Every photo of a position, and of its neighbours in the same chunk, lands in one split"""
    from collections import Counter

    from camera_chess.chessqueries.core import Split
    from camera_chess.chessqueries.data.cvchess import CVChess, assign_splits

    labels = load_labels()
    splits = assign_splits(labels)
    by_fen: dict[str, set] = {}
    for label, split in zip(labels, splits, strict=True):
        by_fen.setdefault(label["gt_fen"], set()).add(split)
    assert all(len(s) == 1 for s in by_fen.values())
    assert dict(Counter(splits)) == dict(CVChess.expected_samples)
    assert set(splits) == {Split.TRAIN, Split.VAL, Split.TEST}
