import chess
import numpy as np
import pytest

from camera_chess.videos import overlay_moves
from camera_chess.videos.chessvision_3lc import SYMBOLS, UNKNOWN
from camera_chess.videos.overlay_moves import candidates, mismatches, placement, track


def test_placement_reads_every_square_in_python_chess_order():
    board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1")
    expected = [0 if (p := board.piece_at(s)) is None else SYMBOLS.index(p.symbol()) for s in chess.SQUARES]
    assert placement(board).tolist() == expected


def test_mismatches_skip_unreadable_squares():
    expected = placement(chess.Board())
    observed = expected.copy()
    observed[[0, 1]] = UNKNOWN
    observed[8] = 0
    assert mismatches(observed, expected) == 1
    assert mismatches(observed, np.stack([expected, expected])).tolist() == [1, 1]


def test_candidates_cover_one_and_two_moves():
    moves, placements = candidates(chess.Board())
    assert len(moves) == 20 + 20 * 20 and placements.shape == (420, 64)
    assert sum(len(m) == 1 for m in moves) == 20


def overlay_of(*ucis):
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return placement(board)


class FakeClassifier:
    """Reads a frame that is already a placement, sure of every square except those marked UNKNOWN"""

    def classify_batch(self, boards):
        log_probs = np.full((*boards.shape, len(SYMBOLS)), np.log(1e-4))
        for observed, out in zip(boards, log_probs):
            known = observed != UNKNOWN
            out[np.flatnonzero(known), observed[known]] = np.log(0.99)
            out[~known] = np.log(1 / len(SYMBOLS))
        return boards, log_probs


def run(monkeypatch, frames):
    """Tracks frames, one every 1 / FPS seconds"""
    timed = [(n / overlay_moves.FPS, f) for n, f in enumerate(frames)]
    monkeypatch.setattr(overlay_moves, "read_frames", lambda path, roi: iter(timed))
    return track("video.mp4", None, FakeClassifier())


def test_follows_a_game_from_stable_frames(monkeypatch):
    start = overlay_of()
    games = run(
        monkeypatch,
        [start] * 2
        + [overlay_of("e2e4")] * 3
        + [overlay_of("e2e4", "e7e5")] * 4
        + [overlay_of("e2e4", "e7e5", "g1f3")] * 3,
    )
    assert len(games) == 1
    game = games[0]
    assert game["start"] == 0.0 and game["end_mismatches"] == 0
    # Each move is timed at the first of its stable frames
    assert game["moves"] == [(0.4, "e2e4"), (1.0, "e7e5"), (1.8, "g1f3")]


def test_two_moves_seen_at_once_are_both_taken(monkeypatch):
    games = run(
        monkeypatch, [overlay_of()] + [overlay_of("d2d4", "d7d5")] * 3 + [overlay_of("d2d4", "d7d5", "c2c4")] * 3
    )
    assert [m for _, m in games[0]["moves"]] == ["d2d4", "d7d5", "c2c4"]


def test_brief_and_unreadable_frames_do_not_move_the_game(monkeypatch):
    e4 = overlay_of("e2e4")
    unreadable = e4.copy()
    unreadable[:10] = UNKNOWN
    wall = np.random.default_rng(0).integers(0, 13, 64)
    frames = [overlay_of()] + [e4] * 2 + [unreadable, wall] + [e4] + [overlay_of("e2e4", "c7c5")] * 3
    games = run(monkeypatch, frames)
    # The unreadable frame and the wall are skipped, so the third e4 frame completes the run
    assert [m for _, m in games[0]["moves"]] == ["e2e4", "c7c5"]


def test_a_start_position_after_a_game_begins_a_new_one(monkeypatch):
    first = [overlay_of()] + [overlay_of("e2e4")] * 3 + [overlay_of("e2e4", "e7e5")] * 3
    # The start position held before the first move doesn't restart the game
    second = [overlay_of()] * 3 + [overlay_of("c2c4")] * 3 + [overlay_of("c2c4", "e7e5")] * 3
    games = run(monkeypatch, first + second)
    assert [[m for _, m in g["moves"]] for g in games] == [["e2e4", "e7e5"], ["c2c4", "e7e5"]]
    assert games[1]["start"] == pytest.approx(7 / overlay_moves.FPS)


def test_end_mismatches_count_a_move_never_confirmed(monkeypatch):
    games = run(
        monkeypatch,
        [overlay_of()]
        + [overlay_of("e2e4")] * 3
        + [overlay_of("e2e4", "e7e5")] * 3
        + [overlay_of("e2e4", "e7e5", "g1f3")],
    )
    assert games[0]["end_mismatches"] == 2


def test_hand_made_pawn_left_comments_are_read_and_kept_when_rewritten(monkeypatch, tmp_path):
    monkeypatch.setattr(overlay_moves, "get_config", lambda: type("Config", (), {"DATA_ROOT": tmp_path})())
    (tmp_path / "game").mkdir()
    moves = ["e2e4", "f7f5", "e4f5", "g7g6", "f5g6", "g8f6", "g6h7", "h8g8", "h7g8q", "f6g8"]
    game = {"moves": [(float(i), uci) for i, uci in enumerate(moves)], "end_mismatches": 0}
    overlay_moves.write_game("game", game)
    pgn = tmp_path / "game" / "gt.pgn"
    pgn.write_text(pgn.read_text().replace("hxg8=Q", "hxg8=Q { [%pawn_left 12.5] Swapped late }"))
    assert overlay_moves.read_pawns_left("game") == {8: 12.5}
    # Read again, the comment stays on the same move
    overlay_moves.write_game("game", game)
    assert overlay_moves.read_pawns_left("game") == {8: 12.5}
    assert "Swapped late" in pgn.read_text()
    # but not on a different one
    game["moves"][8] = (8.0, "h7g8r")
    overlay_moves.write_game("game", game)
    assert overlay_moves.read_pawns_left("game") == {}


def cut_with_keyframes(monkeypatch, keyframes, *args):
    """cut_video's start, and the ffmpeg command it ran, with the video's keyframes at `keyframes`"""
    from camera_chess.videos import overlay_moves

    commands = []
    monkeypatch.setattr(overlay_moves, "keyframes", lambda path: keyframes)
    monkeypatch.setattr(overlay_moves.subprocess, "run", lambda cmd, check: commands.append(cmd))
    start = overlay_moves.cut_video("in.mp4", "out.mp4", *args)
    return start, commands[0]


def test_a_cut_copies_from_the_last_keyframe_before_it(monkeypatch):
    start, cmd = cut_with_keyframes(monkeypatch, [0.04, 5.04, 10.04], 8.6, 20.0)
    assert start == 5.04
    assert cmd[cmd.index("-ss") + 1] == "5.050" and "copy" in cmd and cmd[cmd.index("-t") + 1] == "14.960"


def test_a_cut_skips_a_keyframe_before_the_last_games_last_move(monkeypatch):
    start, _ = cut_with_keyframes(monkeypatch, [0.04, 5.04, 10.04], 8.6, None, 6.0, 12.0)
    assert start == 10.04


def test_a_cut_with_no_keyframe_in_reach_is_re_encoded(monkeypatch):
    start, cmd = cut_with_keyframes(monkeypatch, [0.04, 5.04, 10.04], 8.6, None, 6.0, 9.5)
    assert start == 8.6 and "libx264" in cmd
