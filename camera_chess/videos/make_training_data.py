"""Turns recorded games with a 2D overlay into training data for ChessQueries: crops of the 3D board, each labelled
with the position the overlay shows at that moment.

Each dataset needs, from the usual steps:
  `roi` and `overlay_roi` in video_config.yaml (camera_chess.videos.roi_labeller)
  overlay_moves.json, the true moves and their times (camera_chess.videos.overlay_moves)
  chesscam_probs_<MODEL>.f32.gz and .json, MODEL on every frame (camera_chess.videos.chesscam_model --model MODEL), to
  catch bad labels

Frames are taken between moves, never from just before a move to just after it, when a piece is in the air or in a
hand and the board is neither position. Hands over the board are kept: what they hide is still on the board. Each
position gives up to PER_POSITION frames, spread over the time it stood, and each video up to MAX_FRAMES, so
long games and near-identical frames don't outweigh the other datasets.

A frame whose label the app's model disagrees with on more than MAX_DISAGREEMENTS squares is dropped, as that is a
takeback, a camera cut or a board being reset rather than a hard frame, and a video losing more than MAX_DROPPED of
its frames is left out for a look. Frames the model merely gets a few squares wrong are kept, as those are what the
data is for.

Each crop is exactly the 3D board region, as the app crops the region someone draws. The 2D overlay shows the answer,
so any part of it inside a crop (where it overlaps the board region) is blacked out.

A promotion marked `[%pawn_left]` in gt.pgn (see camera_chess.videos.overlay_moves) is labelled as the pawn left on
its square, until the pawn was swapped for the new piece, or until the square next changes.

The games scored by the tracker replay in CameraChessWeb (data/yunker/benchmark.txt) are never used.

Usage: python -m camera_chess.videos.make_training_data -d yunker/D81hFXZ0Cp4 [-d ...] [-u ...] [--out data/yunker_crops]
-u uses a dataset's labels without MODEL's check, e.g. one an earlier build checked with another model.
       python -m camera_chess.videos.make_training_data --measure-offset -d yunker/LEYxdEWJXCg [-d ...]
Writes the crops to <out>/images/train/ and annotations.json (image, fen, split, group, box, time,
model_agreement, crop_hash), laid out as ChessQueries' ChessCam dataset. The output folder is rebuilt from scratch on each run,
unless --add, which keeps the other datasets there and redoes only these.
"""

import argparse
import gzip
import importlib.util
import json
import shutil
from functools import cache
from pathlib import Path

import chess
import cv2
import decord
import numpy as np
from loguru import logger
from PIL import Image

from camera_chess.config import REPO_ROOT, get_config
from camera_chess.videos.config import load_config, usable, video_path
from camera_chess.videos.overlay_moves import read_pawns_left

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
JPEG_QUALITY = 95
# The model whose readings catch bad labels: the 320 fine-tune of v10, about 2.35x faster to run than the app's 448
MODEL = "chessq-ft-v10-dinov3-320"


@cache
def crop_script():
    """hf/yunker-chess/crop_yunker.py, the Hugging Face dataset's script, whose dhash each crop's crop_hash is"""
    spec = importlib.util.spec_from_file_location("crop_yunker", REPO_ROOT / "hf" / "yunker-chess" / "crop_yunker.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_probs(dataset):
    """(frames, 64, 13) probabilities in FEN order from MODEL, and what they are (frame rate, region)"""
    d = get_config().DATA_ROOT / dataset
    name = f"chesscam_probs_{MODEL}"
    meta = json.loads((d / f"{name}.json").read_text())
    with gzip.open(d / f"{name}.f32.gz", "rb") as f:
        probs = np.frombuffer(f.read(), "<f4").reshape(-1, 64, len(CLASSES))
    return probs, meta


def load_game(dataset):
    """The game in overlay_moves.json, with `pawns_left` from gt.pgn"""
    games = json.loads((get_config().DATA_ROOT / dataset / "overlay_moves.json").read_text())
    if len(games) != 1:
        raise ValueError(f"{dataset} has {len(games)} games; overlay_moves cuts such videos into a dataset per game")
    return {**games[0], "pawns_left": read_pawns_left(dataset)}


def labels(board):
    """Each square's index in CLASSES, in FEN order (a8..h1)"""
    out = np.zeros(64, int)
    for i in range(64):
        piece = board.piece_at(chess.square(i % 8, 7 - i // 8))
        out[i] = 0 if piece is None else CLASSES.index(piece.symbol())
    return out


def positions(game):
    """(board, start, end) for each position after a move, in video seconds, with the board as it stood: a promotion
    in game["pawns_left"] ({move index: second it was swapped, or None}) shows the pawn on its square until then, or
    until a later move changes that square. A position the swap falls in is split in two there."""
    times = [t + OFFSET for t in game["times"]]
    pawns_left = game.get("pawns_left", {})
    board = chess.Board()
    out = []
    # The square a pawn was left on, and when it was swapped (None: not until the square changes)
    pawn, swapped = None, None
    for i, (move, start, end) in enumerate(zip(game["moves"], times, [*times[1:], times[-1] + LEAD_OUT])):
        move = chess.Move.from_uci(move)
        if pawn is not None and pawn in (move.from_square, move.to_square):
            pawn = None
        if i in pawns_left:
            pawn, swapped = move.to_square, pawns_left[i]
        color = board.turn
        board.push(move)
        if pawn is None or (swapped is not None and swapped <= start):
            pawn = None
            out.append((board.copy(), start, end))
            continue
        stood = board.copy()
        # Past the promotion, the square holds the pawn left on it
        owner = color if i in pawns_left else stood.piece_at(pawn).color  # ty: ignore[unresolved-attribute]
        stood.set_piece_at(pawn, chess.Piece(chess.PAWN, owner))
        if swapped is not None and swapped < end:
            out += [(stood, start, swapped), (board.copy(), swapped, end)]
            pawn = None
        else:
            out.append((stood, start, end))
    return out


def frame_times(game):
    """(board, time) of the frames to use: up to PER_POSITION spread over each position's time away from moves, then
    at most MAX_FRAMES spread over the game"""
    chosen = []
    stood = positions(game)
    for i, (board, start, end) in enumerate(stood):
        # The last position has no move after it. A swap of a pawn left on the board is a hand on it, as a move is.
        low = start + AFTER
        high = end - BEFORE if i < len(stood) - 1 else end
        if high - low < MIN_STAND:
            continue
        chosen += [(board, t) for t in np.linspace(low, high, PER_POSITION + 2)[1:-1]]
    if len(chosen) > MAX_FRAMES:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, MAX_FRAMES).round().astype(int)]
    return chosen


def crop_box(roi, width, height):
    """The board region as (x0, y0, x1, y1), kept within the frame"""
    x, y, w, h = roi
    return max(0, x), max(0, y), min(width, x + w), min(height, y + h)


def crop(frame, box, overlay):
    """The frame's crop, with any part of the overlay in it blacked out"""
    x0, y0, x1, y1 = box
    out = frame[y0:y1, x0:x1].copy()
    ox, oy, ow, oh = overlay
    ix0, iy0, ix1, iy1 = max(ox, x0), max(oy, y0), min(ox + ow, x1), min(oy + oh, y1)
    if ix0 < ix1 and iy0 < iy1:
        out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 0
    return out


def frame_index(t, fps, count):
    """The index of the video frame shown `t` seconds in, of `count` frames at `fps`"""
    return min(count - 1, round(t * fps))


def choose(dataset, checked=True):
    """(board, time, model_agreement) of the frames to use from the dataset, or [] if its labels look wrong, and a
    note on how they were checked. Unless `checked` is False, when its labels are used without MODEL's readings
    (model_agreement is then None)."""
    config = load_config(dataset)
    game = load_game(dataset)
    if checked:
        probs, meta = load_probs(dataset)
        if config["roi"] != meta["roi"]:
            raise ValueError(
                f"{dataset}'s chesscam_probs are for another region; rerun camera_chess.videos.chesscam_model "
                f"--model {MODEL}"
            )
    if game.get("end_mismatches", 0) > END_MISMATCHES:
        logger.warning(f"{dataset}: the overlay's last position disagrees with the moves read; skipped")
        return [], "skipped"

    kept, dropped = [], 0
    for board, t in frame_times(game):
        if not checked:
            kept.append((board, t, None))
            continue
        index = frame_index(t, meta["fps"], len(probs))
        agreement = int((probs[index].argmax(-1) == labels(board)).sum())
        if 64 - agreement > MAX_DISAGREEMENTS:
            dropped += 1
            continue
        kept.append((board, t, agreement))
    total = len(kept) + dropped
    if total == 0 or dropped > MAX_DROPPED * total:
        logger.warning(f"{dataset}: the model disagrees with {dropped} of {total} labels; skipped for a look")
        return [], "skipped"
    return kept, f"{dropped} dropped" if checked else "labels not checked"


def make(dataset, out_dir, checked=True):
    """The dataset's annotations, with its crops written to out_dir, or [] if its labels look wrong (see choose)"""
    kept, note = choose(dataset, checked)
    if not kept:
        return []
    config = load_config(dataset)
    reader = decord.VideoReader(str(video_path(dataset)))
    fps = reader.get_avg_fps()
    frames = reader.get_batch([frame_index(t, fps, len(reader)) for _, t, _ in kept]).asnumpy()
    height, width = frames.shape[1:3]
    box = crop_box(config["roi"], width, height)
    x, y, w, h = config["roi"]
    name = dataset.replace("/", "_")
    annotations = []
    for (board, t, agreement), frame in zip(kept, frames):
        image = f"images/train/{name}_{round(t * 100):06d}.jpg"
        cropped = crop(frame, box, config["overlay_roi"])
        Image.fromarray(cropped).save(out_dir / image, quality=JPEG_QUALITY)
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
                # Of the crop before JPEG, as crop_yunker.py hashes the crops it makes
                "crop_hash": crop_script().dhash(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)),
            }
        )
    logger.info(f"{dataset}: {len(annotations)} frames from {len(load_game(dataset)['moves'])} moves ({note})")
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


def benchmark_path():
    return get_config().DATA_ROOT / "yunker" / "benchmark.txt"


def main(datasets, out_dir, unchecked=(), add=False):
    datasets, unchecked = usable(datasets), usable(unchecked)
    benchmark = {f"yunker/{line.strip()}" for line in benchmark_path().read_text().splitlines() if line.strip()}
    if used := [d for d in [*datasets, *unchecked] if d in benchmark or d.split("_game_")[0] in benchmark]:
        raise ValueError(f"{used} are scored by the tracker replay, so must not be trained on")
    kept = []
    if add and (out_dir / "annotations.json").is_file():
        # The other datasets' crops stay; these datasets' old ones go, so a rerun leaves no stale frames behind
        redo = {*datasets, *unchecked}
        for a in json.loads((out_dir / "annotations.json").read_text()):
            if a["group"] in redo:
                (out_dir / a["image"]).unlink(missing_ok=True)
            else:
                kept.append(a)
    elif out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    annotations = [a for dataset in datasets for a in make(dataset, out_dir)]
    annotations += [a for dataset in unchecked for a in make(dataset, out_dir, checked=False)]
    used = len({a["group"] for a in annotations})
    logger.info(f"Wrote {len(annotations)} frames from {used} of {len(datasets) + len(unchecked)} videos to {out_dir}")
    if kept:
        logger.info(f"Kept {len(kept)} frames from {len({a['group'] for a in kept})} other videos already there")
    (out_dir / "annotations.json").write_text(json.dumps(kept + annotations, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", action="append", default=[])
    # e.g. the datasets of an earlier build, already checked by another model, that have no readings from MODEL
    parser.add_argument("--unchecked", "-u", action="append", default=[], help="a dataset to use without MODEL's check")
    # Not data/yunker, which holds the source videos: the output folder is deleted and rebuilt on each run (but --add)
    parser.add_argument("--out", type=Path, default=get_config().DATA_ROOT / "yunker_crops")
    parser.add_argument("--measure-offset", action="store_true")
    parser.add_argument("--add", action="store_true", help="keep the other datasets already in --out")
    args = parser.parse_args()
    if args.measure_offset:
        measure_offset(args.dataset)
    else:
        main(args.dataset, args.out, args.unchecked, args.add)
