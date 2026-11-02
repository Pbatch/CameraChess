"""Turns recorded games with a 2D overlay into training data for ChessQueries: crops of the 3D board, each labelled
with the position the overlay shows at that moment.

Each dataset needs, from the usual steps:
  `roi` and `overlay_roi` in video_config.yaml (camera_chess.videos.roi_labeller)
  overlay_moves.json, the true moves and their times (camera_chess.videos.overlay_moves)
  chesscam_probs.f32.gz and .json, the app's model on every frame (camera_chess.videos.chesscam_model), to catch bad labels

Frames are taken between moves, never from just before a move to just after it, when a piece is in the air or in a
hand and the board is neither position. Hands over the board are kept: what they hide is still on the board. Each
position gives up to PER_POSITION frames, spread over the time it stood, and each video up to MAX_FRAMES, so
long games and near-identical frames don't outweigh the other datasets.

A frame whose label the app's model disagrees with on more than MAX_DISAGREEMENTS squares is dropped, as that is a
takeback, a camera cut or a board being reset rather than a hard frame, and a video losing more than MAX_DROPPED of
its frames is left out for a look. Frames the model merely gets a few squares wrong are kept, as those are what the
data is for.

Each crop is the 3D board region plus MARGIN on every side, as the region someone draws in the app, so training can
crop it again with random margins. The 2D overlay shows the answer, so the top margin stops short of it, and any part
of it still inside a crop (where it overlaps the board region itself) is blacked out.

The games scored by the tracker replay in CameraChessWeb (data/yunker/benchmark.txt) are never used.

Usage: python -m camera_chess.videos.make_training_data -d yunker/D81hFXZ0Cp4 [-d ...] [--out data/yunker_crops]
       python -m camera_chess.videos.make_training_data --measure-offset -d yunker/LEYxdEWJXCg [-d ...]
Writes the crops to <out>/images/train/ and annotations.json (image, fen, split, group, box, time,
model_agreement), laid out as ChessQueries' ChessCam dataset. The output folder is rebuilt from scratch on each run.
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import chess
import decord
import numpy as np
from loguru import logger
from PIL import Image

from camera_chess.config import get_config
from camera_chess.videos.config import load_config, video_path

# Not data/yunker, which holds the source videos: the output folder is deleted and rebuilt on each run
OUT_DIR = get_config().DATA_ROOT / "yunker_crops"
BENCHMARK_PATH = get_config().DATA_ROOT / "yunker" / "benchmark.txt"
# The app model's classes, in the order chesscam_probs holds them
CLASSES = ".PNBRQKpnbrqk"
# Seconds the board changes after the overlay shows a move (negative: before), from --measure-offset: on 3339 moves of
# the 43 benchmark games without takebacks, the app's model read the move 0.3s before the overlay showed it (median;
# quartiles 0.5s to 0.1s before)
OFFSET = -0.3
# Seconds around each move, after OFFSET, whose frames are left out: a hand reaches for the piece before the move and
# lets go of it after
BEFORE = 2.0
AFTER = 0.5
# Seconds after the last move that the final position counts until, as after then the board is being cleared. The
# start position isn't used at all: the videos open with a title drawn over the board, and the start position is in
# every dataset already.
LEAD_OUT = 2.0
# Shortest time a position must stand, outside the windows around moves, for its frames to be used
MIN_STAND = 0.5
PER_POSITION = 3
MAX_FRAMES = 150
MAX_DISAGREEMENTS = 16
# Squares the overlay's last position may differ from the moves read by: a missed move changes at least two, while an
# arrow or highlight can hide one (four benchmark games end one square off and replay fine)
END_MISMATCHES = 1
MAX_DROPPED = 0.2
# Share of the board region's width and height added on every side of a crop
MARGIN = 0.15
JPEG_QUALITY = 95


def load_probs(dataset):
    """(frames, 64, 13) probabilities in FEN order, and what they are (frame rate, region)"""
    d = get_config().DATA_ROOT / dataset
    meta = json.loads((d / "chesscam_probs.json").read_text())
    with gzip.open(d / "chesscam_probs.f32.gz", "rb") as f:
        probs = np.frombuffer(f.read(), "<f4").reshape(-1, 64, len(CLASSES))
    return probs, meta


def load_game(dataset):
    games = json.loads((get_config().DATA_ROOT / dataset / "overlay_moves.json").read_text())
    if len(games) != 1:
        raise ValueError(f"{dataset} has {len(games)} games; overlay_moves cuts such videos into a dataset per game")
    return games[0]


def labels(board):
    """Each square's index in CLASSES, in FEN order (a8..h1)"""
    out = np.zeros(64, int)
    for i in range(64):
        piece = board.piece_at(chess.square(i % 8, 7 - i // 8))
        out[i] = 0 if piece is None else CLASSES.index(piece.symbol())
    return out


def positions(game):
    """(board, start, end) for each position after a move, in video seconds, with the board as it stood"""
    times = [t + OFFSET for t in game["times"]]
    board = chess.Board()
    out = []
    for move, start, end in zip(game["moves"], times, [*times[1:], times[-1] + LEAD_OUT]):
        board.push_uci(move)
        out.append((board.copy(), start, end))
    return out


def frame_times(game):
    """(board, time) of the frames to use: up to PER_POSITION spread over each position's time away from moves, then
    at most MAX_FRAMES spread over the game"""
    chosen = []
    for board, start, end in positions(game):
        # The last position has no move after it
        low = start + AFTER
        high = end - BEFORE if board.ply() < len(game["moves"]) else end
        if high - low < MIN_STAND:
            continue
        chosen += [(board, t) for t in np.linspace(low, high, PER_POSITION + 2)[1:-1]]
    if len(chosen) > MAX_FRAMES:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, MAX_FRAMES).round().astype(int)]
    return chosen


def crop_box(roi, overlay, width, height):
    """The crop around the board region, (x0, y0, x1, y1) within the frame. The overlay sits above the board, so the
    top margin stops at its bottom edge where it would reach into it."""
    x, y, w, h = roi
    dx, dy = MARGIN * w, MARGIN * h
    x0, y0, x1, y1 = (
        max(0, round(x - dx)),
        max(0, round(y - dy)),
        min(width, round(x + w + dx)),
        min(height, round(y + h + dy)),
    )
    ox, oy, ow, oh = overlay
    if ox < x1 and ox + ow > x0 and y0 < oy + oh <= y:
        y0 = oy + oh
    return x0, y0, x1, y1


def crop(frame, box, overlay):
    """The frame's crop, with any part of the overlay in it blacked out"""
    x0, y0, x1, y1 = box
    out = frame[y0:y1, x0:x1].copy()
    ox, oy, ow, oh = overlay
    ix0, iy0, ix1, iy1 = max(ox, x0), max(oy, y0), min(ox + ow, x1), min(oy + oh, y1)
    if ix0 < ix1 and iy0 < iy1:
        out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 0
    return out


def make(dataset, out_dir):
    """The dataset's annotations, with its crops written to out_dir, or [] if its labels look wrong"""
    config = load_config(dataset)
    game = load_game(dataset)
    probs, meta = load_probs(dataset)
    probs_fps = meta["fps"]
    if config["roi"] != meta["roi"]:
        raise ValueError(f"{dataset}'s chesscam_probs are for another region; rerun camera_chess.videos.chesscam_model")
    if game.get("end_mismatches", 0) > END_MISMATCHES:
        logger.warning(f"{dataset}: the overlay's last position disagrees with the moves read; skipped")
        return []

    kept, dropped = [], 0
    for board, t in frame_times(game):
        index = min(len(probs) - 1, round(t * probs_fps))
        agreement = int((probs[index].argmax(-1) == labels(board)).sum())
        if 64 - agreement > MAX_DISAGREEMENTS:
            dropped += 1
            continue
        kept.append((board, t, agreement))
    total = len(kept) + dropped
    if total == 0 or dropped > MAX_DROPPED * total:
        logger.warning(f"{dataset}: the model disagrees with {dropped} of {total} labels; skipped for a look")
        return []

    reader = decord.VideoReader(str(video_path(dataset)))
    fps = reader.get_avg_fps()
    frames = reader.get_batch([min(len(reader) - 1, round(t * fps)) for _, t, _ in kept]).asnumpy()
    height, width = frames.shape[1:3]
    box = crop_box(config["roi"], config["overlay_roi"], width, height)
    x, y, w, h = config["roi"]
    name = dataset.replace("/", "_")
    annotations = []
    for (board, t, agreement), frame in zip(kept, frames):
        image = f"images/train/{name}_{round(t * 100):06d}.jpg"
        Image.fromarray(crop(frame, box, config["overlay_roi"])).save(out_dir / image, quality=JPEG_QUALITY)
        annotations.append(
            {
                "image": image,
                "fen": board.board_fen(),
                "split": "train",
                "group": dataset,
                # The board region within the crop
                "box": [x - box[0], y - box[1], w, h],
                "time": round(t, 2),
                "model_agreement": agreement,
            }
        )
    logger.info(f"{dataset}: {len(annotations)} frames from {len(game['moves'])} moves ({dropped} dropped)")
    return annotations


def measure_offset(datasets):
    """Median seconds from the overlay showing each move to the app's model first reading its squares as after it,
    for two frames in a row, to set OFFSET by"""
    delays = []
    for dataset in datasets:
        game = load_game(dataset)
        probs, meta = load_probs(dataset)
        fps = meta["fps"]
        reads = probs.argmax(-1)
        board = chess.Board()
        for move, t in zip(game["moves"], game["times"]):
            before = labels(board)
            board.push_uci(move)
            after = labels(board)
            changed = np.flatnonzero(before != after)
            # Frames within 3s either side whose changed squares all read as after the move
            first, last = max(0, round((t - 3) * fps)), min(len(reads) - 1, round((t + 3) * fps))
            match = [(reads[i, changed] == after[changed]).all() for i in range(first, last + 1)]
            for i in range(len(match) - 1):
                if match[i] and match[i + 1]:
                    delays.append((first + i) / fps - t)
                    break
    delays = np.array(delays)
    logger.info(
        f"{len(delays)} moves: the model reads the move {np.median(delays):+.2f}s from the overlay (median), "
        f"quartiles {np.percentile(delays, 25):+.2f}s to {np.percentile(delays, 75):+.2f}s"
    )


def main(datasets, out_dir):
    benchmark = {f"yunker/{line.strip()}" for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()}
    if used := [d for d in datasets if d in benchmark or d.split("_game_")[0] in benchmark]:
        raise ValueError(f"{used} are scored by the tracker replay, so must not be trained on")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True)
    annotations = [a for dataset in datasets for a in make(dataset, out_dir)]
    (out_dir / "annotations.json").write_text(json.dumps(annotations, indent=1))
    used = len({a["group"] for a in annotations})
    logger.info(f"Wrote {len(annotations)} frames from {used} of {len(datasets)} videos to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", action="append", required=True)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--measure-offset", action="store_true")
    args = parser.parse_args()
    if args.measure_offset:
        measure_offset(args.dataset)
    else:
        main(args.dataset, args.out)
