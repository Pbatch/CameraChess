"""Reads a game's moves from the 2D board some videos draw over the camera view, for use as ground truth.

Each video needs `overlay_roi` ([x, y, width, height] of the 2D board, white at the bottom) in video_config.yaml,
which camera_chess.roi_labeller draws. Each frame's squares are read by ChessVision-3LC's square classifier (see
camera_chess.chessvision_3lc). The game is followed by accepting the legal move (or pair of moves) that best explains
the overlay, by the summed log-probability of every square, for several frames in a row. Scoring with probabilities,
not hard labels, stops one confidently misread piece from picking the wrong move.

Only frames that a reachable position explains are used: the current one, one or two legal moves on, or the start
position. Anything else, such as the camera view or the wall once the overlay is taken away, is ignored.

Usage: python -m camera_chess.overlay_moves -d yunker/Br8Y3kCJRCs [-d ...]
Writes gt.pgn and overlay_moves.json (with each move's time) to the dataset. A new game starts whenever the start
position comes back, and a video of several games is cut into a dataset per game, `<dataset>_game_N` (see write).
"""

import argparse
import copy
import json
import os
import subprocess

import chess
import chess.pgn
import numpy as np
import yaml
from loguru import logger

from camera_chess.chessvision_3lc import SQUARE, SYMBOLS, UNKNOWN, SquareClassifier
from camera_chess.constants import DATA_DIR

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


def load_config(dataset):
    with open(os.path.join(DATA_DIR, "video_config.yaml")) as f:
        return yaml.safe_load(f)[dataset]


def video_path(dataset):
    d = os.path.join(DATA_DIR, dataset)
    return os.path.join(d, next(p for p in sorted(os.listdir(d)) if p.startswith("video.") and not p.endswith(".json")))


def read_frames(path, roi):
    """Yields (time, SIZE x SIZE x 3 crop of the overlay) at FPS"""
    x, y, w, h = roi
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        # Decoding is most of the cost, and these videos' B-frames (over half) aren't needed by other frames, so they
        # are skipped. The fps filter takes the nearest decoded frame instead, at most a frame or two away.
        "-skip_frame",
        "noref",
        "-i",
        path,
        "-vf",
        f"fps={FPS},crop={w}:{h}:{x}:{y},scale={SIZE}:{SIZE}:flags=bilinear",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert process.stdout is not None
    n = 0
    while True:
        raw = process.stdout.read(SIZE * SIZE * 3)
        if len(raw) < SIZE * SIZE * 3:
            break
        yield n / FPS, np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)
        n += 1
    process.wait()


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

    for t, frame in read_frames(path, roi):
        observed, log_probs = classifier.classify(frame)
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


def cut_video(path, out_path, start, end):
    """Re-encodes [start, end) seconds of the video, as copying the stream could only cut at keyframes. end None is
    the end of the video."""
    duration = [] if end is None else ["-t", str(end - start)]
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", path, *duration, "-an", "-c:v", "libx264"]
    subprocess.run([*cmd, "-crf", "18", "-preset", "fast", out_path], check=True)


def write_game(dataset, game):
    d = os.path.join(DATA_DIR, dataset)
    pgn = chess.pgn.Game()
    node = pgn
    for _, uci in game["moves"]:
        node = node.add_variation(chess.Move.from_uci(uci))
    pgn.headers["Result"] = node.board().result()
    with open(os.path.join(d, "gt.pgn"), "w") as f:
        f.write(str(pgn) + "\n")
    with open(os.path.join(d, "overlay_moves.json"), "w") as f:
        json.dump(
            [
                {
                    "times": [t for t, _ in game["moves"]],
                    "moves": [m for _, m in game["moves"]],
                    "end_mismatches": game["end_mismatches"],
                }
            ],
            f,
            indent=1,
        )


def write(dataset, games):
    """Writes gt.pgn and overlay_moves.json to the dataset. A video of several games is instead cut into a dataset per
    game, `<dataset>_game_N`, from LEAD_IN before its start position is first seen (but after the last game's last
    move) to where the next part starts, with its times from the cut and its own entry in video_config.yaml, as the
    tracker follows one game from the video's start."""
    if len(games) == 1:
        write_game(dataset, games[0])
        return
    config_path = os.path.join(DATA_DIR, "video_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cuts = [
        max(game["start"] - LEAD_IN, prev["moves"][-1][0] if prev else 0) for prev, game in zip([None, *games], games)
    ]
    for n, game in enumerate(games, 1):
        part = f"{dataset}_game_{n}"
        start = round(cuts[n - 1], 1)
        end = round(cuts[n], 1) if n < len(games) else None
        os.makedirs(os.path.join(DATA_DIR, part), exist_ok=True)
        logger.info(f"Cutting {start}s to {end if end is not None else 'the end'}s of {dataset} into {part}")
        cut_video(video_path(dataset), os.path.join(DATA_DIR, part, "video.mp4"), start, end)
        moves = [(round(t - start, 1), uci) for t, uci in game["moves"]]
        write_game(part, {**game, "moves": moves})
        # A deep copy, as safe_dump writes lists shared between entries as aliases
        config[part] = copy.deepcopy(config[dataset])
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


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
