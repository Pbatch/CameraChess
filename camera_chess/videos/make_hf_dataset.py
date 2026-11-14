"""Builds the YunkerChess dataset for Hugging Face (hf/yunker-chess): each labelled frame as a YouTube video, a frame
of it, the board region to crop and the FEN, with no images, so the crops can be made again from the videos with
hf/yunker-chess/crop_yunker.py. Beside the frames, a table of the games, with each move and when it was played.

train and validation: the frames in data/yunker_crops/annotations.json (camera_chess.videos.make_training_data), as
trained on, with VALIDATION of them held out by whole video.
test: the games scored by the tracker replay (data/yunker/benchmark.txt), which are never trained on, with their
frames chosen and checked against MODEL as make_training_data chooses them.

Frames are given in the YouTube video itself. A video of several games is cut into a clip per game,
`<id>_game_N` (camera_chess.videos.overlay_moves), whose times start at the cut. Where the cut started isn't kept,
so it is found by matching the clip's frames to the video's, and cached in data/yunker/cut_starts.json.

Each frame has a hash of its crop (crop_yunker.dhash), so crop_yunker.py can tell when a download gives other crops:
make_training_data's crop_hash for train and validation, and of the crop decoded from the video for test.

Usage: python -m camera_chess.videos.make_hf_dataset [--crops data/yunker_crops] [--out data/hf_yunker]
Writes <out>/data/{train,validation,test}.jsonl and <out>/games/{train,validation,test}.jsonl, with the card
(README.md) and crop_yunker.py from hf/yunker-chess beside them.
"""

import argparse
import hashlib
import json
import multiprocessing
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import chess
import cv2
import decord
import numpy as np
from loguru import logger
from PIL import Image

from camera_chess.config import REPO_ROOT, get_config
from camera_chess.videos.config import load_all, video_path
from camera_chess.videos.make_training_data import (
    OFFSET,
    benchmark_path,
    choose,
    crop_box,
    crop_script,
    frame_index,
    load_game,
    positions,
)

CARD_DIR = REPO_ROOT / "hf" / "yunker-chess"
# Thumbnails a clip's frames are matched to its video's by
THUMB = (32, 18)
# Mean difference (0-255) of the best match, above which it isn't trusted. Copied frames match to about 0.2, and the
# next frame along is 0.4 or more off.
MAX_MATCH_ERROR = 0.3
# The share of the trained-on frames held out for validation, by whole video
VALIDATION = 0.1
SPLITS = ("train", "validation", "test")
# Processes the videos are decoded and the crops read in: each holds a batch of DECODE_BATCH frames at once
WORKERS = 8
DECODE_BATCH = 32


def parent(group):
    """yunker/<id>_game_N's video, yunker/<id>"""
    return group.split("_game_")[0]


def game_number(group):
    return int(group.split("_game_")[1]) if "_game_" in group else 1


@cache
def video_info(dataset):
    """(frame count, frames a second, width, height) of the dataset's video, as decord reads it. OpenCV reads them from
    the container about 10x faster than decord opens the video, and agreed with decord on all 806 Yunker videos; decord
    is the fallback for a container that doesn't give the frame count."""
    capture = cv2.VideoCapture(str(video_path(dataset)))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if count > 0 and fps > 0:
        return count, fps, width, height
    reader = decord.VideoReader(str(video_path(dataset)))
    height, width = reader[0].shape[:2]
    return len(reader), reader.get_avg_fps(), width, height


def thumbnails(path):
    """(frames, pixels) small grey thumbnails of every frame of the video"""
    w, h = THUMB
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-fps_mode", "passthrough"]
    out = subprocess.run(
        [*cmd, "-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, w * h).astype(np.float32)


def match_start(clip, video):
    """(frame of `video` that `clip`'s first frame is, mean difference there) by matching 60 of the clip's frames"""
    if len(clip) > len(video):
        raise ValueError("the clip is longer than the video")
    sampled = np.linspace(min(50, len(clip) - 1), len(clip) - 1, 60).round().astype(int)
    starts = np.arange(len(video) - sampled.max())
    errors = np.mean([np.abs(video[starts + j] - clip[j]).mean(1) for j in sampled], 0)
    best = int(np.argmin(errors))
    return int(starts[best]), float(errors[best])


def cut_starts(groups):
    """{group: the frame of its YouTube video that its first frame is}, 0 unless it is a clip cut from the video"""
    cache_path = get_config().DATA_ROOT / "yunker" / "cut_starts.json"
    cached = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    out = {}
    by_parent = {}
    for group in groups:
        if "_game_" not in group:
            out[group] = 0
            continue
        # Keyed by the clip's size too, so a clip cut again is matched again
        size = video_path(group).stat().st_size
        if (entry := cached.get(group)) and entry["size"] == size:
            out[group] = entry["frame"]
        else:
            by_parent.setdefault(parent(group), []).append((group, size))
    for video, clips in sorted(by_parent.items()):
        frames = thumbnails(video_path(video))
        for group, size in clips:
            start, error = match_start(thumbnails(video_path(group)), frames)
            if error > MAX_MATCH_ERROR:
                raise ValueError(f"{group} matches {video} at frame {start} only to {error:.2f}")
            logger.info(f"{group} starts at frame {start} of {video} (difference {error:.3f})")
            out[group] = start
            cached[group] = {"frame": start, "size": size}
            cache_path.write_text(json.dumps(cached, indent=1, sort_keys=True))
    return out


def benchmark_frames():
    """make_training_data's annotations for the benchmark games, without writing crops: {group, time, fen,
    model_agreement}. A video of several games is in benchmark.txt both whole and as its clips; only the clips,
    with one game each, are used."""
    out = []
    for group in [f"yunker/{line.strip()}" for line in benchmark_path().read_text().splitlines() if line.strip()]:
        moves_path = get_config().DATA_ROOT / group / "overlay_moves.json"
        if not moves_path.is_file():
            logger.warning(f"{group}: no moves read from it; left out")
            continue
        if len(json.loads(moves_path.read_text())) != 1:
            continue
        kept, note = choose(group)
        logger.info(f"{group}: {len(kept)} frames from {len(load_game(group)['moves'])} moves ({note})")
        out += [
            {"group": group, "time": t, "fen": board.board_fen(), "model_agreement": agreement}
            for board, t, agreement in kept
        ]
    return out


def pool():
    # Spawned, not forked, as decord's threads don't survive a fork
    return ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"))


def image_sizes(paths):
    """(width, height) of each image, read from its header"""
    out = []
    for path in paths:
        with Image.open(path) as image:
            out.append(image.size)
    return out


def check_crops(annotations, crops_dir, config):
    """Raises if a crop isn't the size of its group's board region now, as then video_config.yaml has changed since
    the crops were made, and the rows would give other crops"""
    by_group = {}
    for a in annotations:
        by_group.setdefault(a["group"], []).append(crops_dir / a["image"])
    wrong = Counter()
    with pool() as executor:
        for group, sizes in zip(by_group, executor.map(image_sizes, by_group.values(), chunksize=8)):
            _, _, width, height = video_info(group)
            x0, y0, x1, y1 = crop_box(config[group]["roi"], width, height)
            wrong[group] = sum(size != (x1 - x0, y1 - y0) for size in sizes)
    if wrong := +wrong:
        raise ValueError(f"crops not of their board region in video_config.yaml: {dict(wrong)}")


def decoded_hashes(path, frames, box, overlay):
    """crop_yunker.dhash of the crop `box` ([x, y, width, height]) of each of the video's frames"""
    script = crop_script()
    reader = decord.VideoReader(str(path))
    out = []
    for i in range(0, len(frames), DECODE_BATCH):
        for frame in reader.get_batch(frames[i : i + DECODE_BATCH]).asnumpy():
            out.append(script.dhash(script.crop(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), box, overlay)))
    return out


def crop_hashes(annotations, config):
    """Each annotation's crop_yunker.dhash: its crop_hash if make_training_data gave it one, else of its crop from
    its video"""
    out = [a.get("crop_hash") for a in annotations]
    to_decode = {}
    for i, a in enumerate(annotations):
        if out[i] is None:
            to_decode.setdefault(a["group"], []).append(i)
    jobs = []
    for group, indices in to_decode.items():
        count, fps, width, height = video_info(group)
        x0, y0, x1, y1 = crop_box(config[group]["roi"], width, height)
        frames = [frame_index(annotations[i]["time"], fps, count) for i in indices]
        jobs.append((video_path(group), frames, [x0, y0, x1 - x0, y1 - y0], config[group]["overlay_roi"]))
    if jobs:
        with pool() as executor:
            for indices, hashes in zip(to_decode.values(), executor.map(decoded_hashes, *zip(*jobs))):
                for i, h in zip(indices, hashes):
                    out[i] = h
    return out


@cache
def game_positions(group):
    """(the seconds the board changed at each move, [(ply, placement, start, end) of each position]) of the group's
    game, in the group's video seconds (see positions)"""
    game = load_game(group)
    changes = [m + OFFSET for m in game["times"]]
    return changes, [(len(board.move_stack), board.board_fen(), start, end) for board, start, end in positions(game)]


def move_context(group, t, fen):
    """(ply, seconds since the board changed to this position, seconds until it next changes or None) of a frame"""
    changes, placements = game_positions(group)
    for ply, placement, start, end in placements:
        if start <= t <= end and placement == fen:
            until = round(changes[ply] - t, 2) if ply < len(changes) else None
            return ply, round(t - changes[ply - 1], 2), until
    raise ValueError(f"{group}: no position {fen} at {t}s; its overlay_moves.json changed since it was labelled")


def rows(annotations, config, hashes):
    """The dataset's rows, in the YouTube videos' frames, sorted by video and frame"""
    starts = cut_starts(sorted({a["group"] for a in annotations}))
    out = []
    for a, crop_hash in zip(annotations, hashes):
        group = a["group"]
        count, fps, _, _ = video_info(group)
        video_count, video_fps, width, height = video_info(parent(group))
        # decord averages a clip's rate over the clip, so a cut clip's can be a little off its video's
        if abs(fps - video_fps) > 1e-3 * video_fps:
            raise ValueError(f"{group} is at {fps} frames a second, its video at {video_fps}")
        frame = starts[group] + frame_index(a["time"], fps, count)
        assert frame < video_count
        # The region cropped: the board region cut off at the frame's edge
        x0, y0, x1, y1 = crop_box(config[group]["roi"], width, height)
        ply, since_move, until_move = move_context(group, a["time"], a["fen"])
        video_id = parent(group).removeprefix("yunker/")
        out.append(
            {
                "id": f"{video_id}_{frame:06d}",
                "video_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "game": game_number(group),
                "frame": frame,
                "time": round(frame / video_fps, 3),
                "fen": a["fen"],
                "ply": ply,
                "since_move": since_move,
                "until_move": until_move,
                "board_box": [x0, y0, x1 - x0, y1 - y0],
                "overlay_box": config[group]["overlay_roi"],
                "video_width": width,
                "video_height": height,
                "fps": round(video_fps, 6),
                "model_agreement": a.get("model_agreement"),
                "crop_hash": crop_hash,
            }
        )
    out.sort(key=lambda r: (r["video_id"], r["frame"]))
    if duplicates := [i for i, n in Counter(r["id"] for r in out).items() if n > 1]:
        raise ValueError(f"frames labelled twice: {duplicates[:5]}")
    return out


def hold_out(records, share):
    """(kept, held out): whole videos, in an order set by their ids, held out until they have `share` of the frames"""
    by_video = Counter(r["video_id"] for r in records)
    order = sorted(by_video, key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    held, frames = set(), 0
    for video in order:
        if frames >= share * len(records):
            break
        held.add(video)
        frames += by_video[video]
    return [r for r in records if r["video_id"] not in held], [r for r in records if r["video_id"] in held]


def game_rows(groups, starts):
    """Each group's game: its moves, and when the 2D overlay showed each, in the YouTube video"""
    out = []
    for group in sorted(groups):
        game = load_game(group)
        _, fps, _, _ = video_info(parent(group))
        video_id = parent(group).removeprefix("yunker/")
        frames = [starts[group] + round(t * fps) for t in game["times"]]
        board = chess.Board()
        sans = []
        for uci in game["moves"]:
            move = chess.Move.from_uci(uci)
            sans.append(board.san(move))
            board.push(move)
        out.append(
            {
                "id": f"{video_id}_game_{game_number(group)}",
                "video_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "game": game_number(group),
                "moves": game["moves"],
                "san": sans,
                "move_frames": frames,
                "move_times": [round(f / fps, 3) for f in frames],
                "final_fen": board.board_fen(),
                # Promotions whose pawn was left on the board: when it was swapped for the new piece, in the YouTube
                # video, or None if it stood until its square next changed
                "pawns_left": [
                    {"ply": i + 1, "swapped_at": None if s is None else round(starts[group] / fps + s, 3)}
                    for i, s in sorted(game["pawns_left"].items())
                ],
            }
        )
    return out


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main(crops_dir, out_dir):
    config = load_all()
    trained = json.loads((crops_dir / "annotations.json").read_text())
    if unhashed := sorted({a["group"] for a in trained if "crop_hash" not in a}):
        raise ValueError(
            f"{len(unhashed)} videos' annotations have no crop_hash, e.g. {unhashed[0]}; rerun "
            "camera_chess.videos.make_training_data on them (--add keeps the others)"
        )
    check_crops(trained, crops_dir, config)
    test = benchmark_frames()
    train, validation = hold_out(rows(trained, config, crop_hashes(trained, config)), VALIDATION)
    splits = {"train": train, "validation": validation, "test": rows(test, config, crop_hashes(test, config))}
    videos = {split: {r["video_id"] for r in records} for split, records in splits.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if shared := videos[a] & videos[b]:
            raise ValueError(f"videos in both {a} and {b}: {sorted(shared)}")
    groups = {a["group"] for a in [*trained, *test]}
    starts = cut_starts(sorted(groups))
    by_id = {(parent(g).removeprefix("yunker/"), game_number(g)): g for g in groups}
    for split, records in splits.items():
        write_jsonl(out_dir / "data" / f"{split}.jsonl", records)
        games = sorted({(r["video_id"], r["game"]) for r in records})
        write_jsonl(out_dir / "games" / f"{split}.jsonl", game_rows([by_id[g] for g in games], starts))
        positions_ = len({r["fen"] for r in records})
        logger.info(
            f"{split}: {len(records)} frames, {positions_} positions, {len(games)} games from "
            f"{len(videos[split])} videos"
        )
    for name in ("README.md", "crop_yunker.py"):
        shutil.copy(CARD_DIR / name, out_dir / name)
    logger.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, default=get_config().yunker_root, help="make_training_data's --out")
    parser.add_argument("--out", type=Path, default=get_config().DATA_ROOT / "hf_yunker")
    args = parser.parse_args()
    main(args.crops, args.out)
