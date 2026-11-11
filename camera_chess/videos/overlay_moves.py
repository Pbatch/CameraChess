"""Reads a game's moves from the 2D board some videos draw over the camera view, for use as ground truth.

Each video needs `overlay_roi` ([x, y, width, height] of the 2D board, white at the bottom) in video_config.yaml,
which camera_chess.videos.roi_labeller draws. Each frame's squares are read by ChessVision-3LC's square classifier (see
camera_chess.videos.chessvision_3lc). The game is followed by accepting the legal move (or pair of moves) that best explains
the overlay, by the summed log-probability of every square, for several frames in a row. Scoring with probabilities,
not hard labels, stops one confidently misread piece from picking the wrong move.

Only frames that a reachable position explains are used: the current one, one or two legal moves on, or the start
position. Anything else, such as the camera view or the wall once the overlay is taken away, is ignored.

Usage: python -m camera_chess.videos.overlay_moves -d yunker/Br8Y3kCJRCs [-d ...]
Writes gt.pgn and overlay_moves.json (with each move's time) to the dataset. A new game starts whenever the start
position comes back, and a video of several games is cut into a dataset per game, `<dataset>_game_N` (see write).

A promotion whose pawn the players left on the board, as blitz players often do, is marked by hand in gt.pgn with a
comment on the move: `{ [%pawn_left] }` if the pawn stood there until the square next changed, or
`{ [%pawn_left 433.0] }` if it was swapped for the new piece at that second of the video. The overlay shows the new
piece either way, so it stays the true move, but camera_chess.videos.make_training_data labels the square as the pawn.
Rerunning this keeps such comments on moves that are read the same again.
"""

import argparse
import copy
import itertools
import json
import queue
import re
import subprocess
import threading

import chess
import chess.pgn
import numpy as np
from loguru import logger

from camera_chess.config import get_config
from camera_chess.videos import ffmpeg
from camera_chess.videos.chessvision_3lc import BATCH, SQUARE, SYMBOLS, UNKNOWN, SquareClassifier
from camera_chess.videos.config import load_all, load_config, save_all, video_path

SIZE = 8 * SQUARE
# Frames read per second of video
FPS = 5
# Frames in a row a position must be seen for before it is accepted
STABLE = 3
# Log-likelihood a candidate must beat the current position by
MARGIN = 4.0
# Readable squares the accepted position may disagree on
MAX_MISMATCHES = 3
# Log-likelihood each extra move costs, so a pair that only adds a reply on unreadable squares loses
MOVE_COST = 2.0
# Start-position squares that may disagree, as arrows and highlights can hide a few
START_MISMATCHES = 2
# Seconds a game's part of a video starts before its start position is seen, as the overlay can be hidden while the
# board is set up and only come back as the first move is played
LEAD_IN = 3.0


# Frames decoded ahead of the classifier
READ_AHEAD = 32


def read_frames(path, roi):
    """Yields (time, SIZE x SIZE x 3 crop of the overlay) at FPS, read on a thread, so ffmpeg keeps decoding while the
    classifier runs instead of waiting on a full pipe (a frame is far bigger than the pipe's buffer)"""
    frames = queue.Queue(maxsize=READ_AHEAD)
    errors = []

    def read():
        try:
            for item in ffmpeg.read_frames(path, roi, SIZE, FPS):
                frames.put(item)
        except Exception as error:  # noqa: BLE001 - re-raised below, on the caller's thread
            errors.append(error)
        finally:
            frames.put(None)

    threading.Thread(target=read, daemon=True).start()
    while (item := frames.get()) is not None:
        yield item
    if errors:
        raise errors[0]


def classified(path, roi, classifier):
    """Yields (time, each square's label, log-probabilities) for each frame, classified BATCH frames at a time"""
    for batch in itertools.batched(read_frames(path, roi), BATCH):
        times, frames = zip(*batch)
        labels, log_probs = classifier.classify_batch(np.stack(frames))
        yield from zip(times, labels, log_probs)


# Frames with fewer readable squares than this are skipped, as a frame of unreadable squares matches any position
MIN_KNOWN = 56


# The piece of each symbol after '.', for reading a board's bitboards
PIECES = [chess.Piece.from_symbol(symbol) for symbol in SYMBOLS[1:]]


def placement(board):
    """Each square's index in SYMBOLS, from the board's 12 piece bitboards at once, as reading 64 squares one by one
    is slow when it is done for every candidate"""
    masks = np.array([board.pieces_mask(p.piece_type, p.color) for p in PIECES], dtype="<u8")
    bits = np.unpackbits(masks.view(np.uint8).reshape(12, 8), axis=1, bitorder="little")
    return (bits * np.arange(1, 13)[:, None]).sum(axis=0)


def mismatches(observed, expected):
    """Squares, of those readable, where the observed placement differs from each expected one"""
    return ((observed != expected) & (observed != UNKNOWN)).sum(axis=-1)


def candidates(board):
    """Move lists one or two moves on, and their placements as an (N, 64) array"""
    moves, placements = [], []
    for move1 in board.legal_moves:
        board.push(move1)
        moves.append([move1])
        placements.append(placement(board))
        for move2 in board.legal_moves:
            board.push(move2)
            moves.append([move1, move2])
            placements.append(placement(board))
            board.pop()
        board.pop()
    return moves, np.array(placements, dtype=int).reshape(-1, 64)


def track(path, roi, classifier):
    """Games from the overlay, starting a new one whenever the start position comes back. Each game has the `start`
    time its start position was first seen, its moves as (time, uci) and `end_mismatches`, the squares its last
    explained frame disagrees with the final position on, which is 0 unless a move was still waiting to be seen for
    long enough."""
    start = placement(chess.Board())
    games: list[dict] = []
    board = None
    moves, placements, costs = [], np.zeros((0, 64), dtype=int), np.zeros(0)
    last = None
    # The candidate seen on the last frames, and for how many frames in a row
    pending, count = None, 0

    def finish():
        if games:
            games[-1]["end_mismatches"] = int(mismatches(last, placement(board)))

    for t, observed, log_probs in classified(path, roi, classifier):
        if (observed != UNKNOWN).sum() < MIN_KNOWN:
            continue
        if mismatches(observed, start) <= START_MISMATCHES and (board is None or len(board.move_stack) > 1):
            finish()
            logger.info(f"Game {len(games) + 1} starts at {t:.1f}s")
            games.append({"start": round(t, 1), "moves": []})
            board = chess.Board()
            moves, placements = candidates(board)
            costs = MOVE_COST * np.array([len(m) for m in moves])
            pending, count = None, 0
            continue
        if board is None:
            continue
        # Only frames a reachable position explains count, so others don't end a move's run of frames or the game
        off = mismatches(observed, placements)
        if mismatches(observed, placement(board)) > MAX_MISMATCHES and not (off <= MAX_MISMATCHES).any():
            continue
        last = observed
        # The game is over
        if not moves:
            continue
        current = log_probs[np.arange(64), placement(board)].sum()
        scores = log_probs[np.arange(64), placements].sum(axis=1) - costs
        index = int(scores.argmax())
        if scores[index] < current + MARGIN or off[index] > MAX_MISMATCHES:
            pending, count = None, 0
            continue
        key = tuple(m.uci() for m in moves[index])
        pending, count = (key, count + 1) if key == pending else (key, 1)
        if count >= STABLE:
            for move in moves[index]:
                games[-1]["moves"].append((round(t - (STABLE - 1) / FPS, 1), move.uci()))
                board.push(move)
            moves, placements = candidates(board)
            costs = MOVE_COST * np.array([len(m) for m in moves])
            pending, count = None, 0
    finish()
    # A start position seen again after a single move is a reset, not a game
    return [game for game in games if len(game["moves"]) > 1]


def keyframes(path):
    """The video's keyframe timestamps in seconds, which read_frames' times and ffmpeg's -ss both count in (the first
    frame's isn't always 0)"""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries", "packet=pts_time,flags"]
    packets = json.loads(subprocess.run([*cmd, path], check=True, capture_output=True, text=True).stdout)["packets"]
    return sorted(float(p["pts_time"]) for p in packets if "K" in p.get("flags", ""))


def cut_video(path, out_path, start, end, earliest=0.0, latest=None):
    """Copies the video from a keyframe up to `end` seconds (None: the end of the video) and returns the second it
    starts at, as copying the stream can only start at a keyframe: the last at or before `start` that is no earlier than
    `earliest`, else the first after `start` and no later than `latest`. With neither (keyframes are 5s apart in the
    Yunker videos) it re-encodes from `start`, which takes about a minute for ten minutes of 720p."""
    times = keyframes(path)
    before = [k for k in times if earliest <= k <= start]
    after = [k for k in times if start < k and (latest is None or k <= latest)]
    duration = lambda s: [] if end is None else ["-t", f"{end - s:.3f}"]
    if before or after:
        s = before[-1] if before else after[0]
        # Input seeking with a copied stream starts at the last keyframe at or before -ss; the margin keeps rounding
        # from landing on the one before
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{s + 0.01:.3f}", "-i", path, *duration(s), "-an"]
        subprocess.run([*cmd, "-c:v", "copy", "-avoid_negative_ts", "make_zero", out_path], check=True)
        return s
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", path, *duration(start), "-an", "-c:v", "libx264"]
    subprocess.run([*cmd, "-crf", "18", "-preset", "fast", out_path], check=True)
    return start


PAWN_LEFT = re.compile(r"\[%pawn_left(?:\s+([\d.]+))?\]")


def read_comments(path):
    """Each move's comment in a gt.pgn, by its index in the game, as (uci, comment)"""
    if not path.exists():
        return {}
    with path.open() as f:
        game = chess.pgn.read_game(f)
    if game is None:
        return {}
    return {i: (node.move.uci(), node.comment) for i, node in enumerate(game.mainline()) if node.comment}


def read_pawns_left(dataset):
    """The promotions whose pawn was left on the board, by move index: the second of the video the pawn was swapped
    for the new piece, or None if it stood until the square next changed"""
    out = {}
    for i, (uci, comment) in read_comments(get_config().DATA_ROOT / dataset / "gt.pgn").items():
        if (match := PAWN_LEFT.search(comment)) is not None:
            if len(uci) != 5:
                raise ValueError(f"{dataset}: move {i + 1}, {uci}, is marked [%pawn_left] but isn't a promotion")
            out[i] = None if match.group(1) is None else float(match.group(1))
    return out


def write_game(dataset, game):
    d = get_config().DATA_ROOT / dataset
    # Comments added by hand, kept on the moves that are read the same again
    comments = read_comments(d / "gt.pgn")
    pgn = chess.pgn.Game()
    node = pgn
    for i, (_, uci) in enumerate(game["moves"]):
        node = node.add_variation(chess.Move.from_uci(uci))
        if i in comments and comments[i][0] == uci:
            node.comment = comments[i][1]
    pgn.headers["Result"] = node.board().result()
    (d / "gt.pgn").write_text(str(pgn) + "\n")
    (d / "overlay_moves.json").write_text(
        json.dumps(
            [
                {
                    "times": [t for t, _ in game["moves"]],
                    "moves": [m for _, m in game["moves"]],
                    "end_mismatches": game["end_mismatches"],
                }
            ],
            indent=1,
        )
    )


def write(dataset, games):
    """Writes gt.pgn and overlay_moves.json to the dataset. A video of several games is instead cut into a dataset per
    game, `<dataset>_game_N`, from about LEAD_IN before its start position is first seen (the keyframe nearest it, but
    after the last game's last move; see cut_video) to where the next part starts, with its times from the cut and its own entry in video_config.yaml, so each
    dataset is one game from its video's start."""
    if len(games) == 1:
        write_game(dataset, games[0])
        return
    config = load_config(dataset)
    parts = {}
    cuts = [
        max(game["start"] - LEAD_IN, prev["moves"][-1][0] if prev else 0) for prev, game in zip([None, *games], games)
    ]
    for n, (prev, game) in enumerate(zip([None, *games], games), 1):
        part = f"{dataset}_game_{n}"
        end = round(cuts[n], 1) if n < len(games) else None
        part_dir = get_config().DATA_ROOT / part
        part_dir.mkdir(parents=True, exist_ok=True)
        # From a keyframe after the last game's last move, and at least a second before this game's first
        earliest = prev["moves"][-1][0] if prev else 0.0
        start = cut_video(
            video_path(dataset), part_dir / "video.mp4", cuts[n - 1], end, earliest, game["moves"][0][0] - 1
        )
        logger.info(f"Cut {start:.2f}s to {end if end is not None else 'the end'}s of {dataset} into {part}")
        moves = [(round(t - start, 1), uci) for t, uci in game["moves"]]
        write_game(part, {**game, "moves": moves})
        # A deep copy, as safe_dump writes lists shared between entries as aliases
        parts[part] = copy.deepcopy(config)
    # Read again, as video_config.yaml may have changed while the video was cut (e.g. boxes saved by roi_labeller)
    save_all(load_all() | parts)


def main(datasets):
    for dataset in datasets:
        config = load_config(dataset)
        path = video_path(dataset)
        classifier = SquareClassifier()
        games = track(path, config["overlay_roi"], classifier)
        write(dataset, games)
        for i, game in enumerate(games):
            board = chess.Board()
            sans = []
            for _, uci in game["moves"]:
                move = chess.Move.from_uci(uci)
                sans.append(board.san(move))
                board.push(move)
            logger.info(
                f"{dataset} game {i + 1}: {len(sans)} plies, last at {game['moves'][-1][0]}s, "
                f"{game['end_mismatches']} squares off the overlay at the end: {' '.join(sans)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", action="append", required=True)
    args = parser.parse_args()
    main(args.dataset)
