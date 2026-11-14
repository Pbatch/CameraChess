# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pinned to what the dataset's crop_hash values were made with, as other decoders and resizers give other pixels
#     "av==19.0.2",
#     "opencv-python-headless==4.11.0.86",
#     "numpy>=2,<3",
#     # Not pinned: YouTube's changes soon break older releases
#     "yt-dlp>=2026.10.26",
# ]
# ///
"""Makes the YunkerChess crops: downloads each video from YouTube, crops the board region from each labelled frame
(blacking out any part of the 2D overlay inside it, as it shows the answer) and writes them with their FENs.

    uv run crop_yunker.py --out yunker_crops                   # every split, rows from the Hugging Face Hub
    uv run crop_yunker.py --out yunker_crops --split test      # just the test split
    uv run crop_yunker.py --out yunker_crops --rows path/to/yunker-chess --video Br8Y3kCJRCs

Writes <out>/<split>/<id>.jpg and <out>/<split>/metadata.jsonl (file_name, fen, video_id, game, frame, time, ply,
since_move, until_move), so the output loads with datasets.load_dataset("imagefolder", data_dir=<out>). Videos
already cropped are skipped, so a stopped run can be started again. Each video is deleted once cropped, unless
--keep-videos.

Each crop's hash is compared with the dataset's (crop_hash), and a video whose crops mostly differ is reported: a
download of another encoding than the dataset's, or frames out of step with it.

Needs ffmpeg on the PATH for yt-dlp. A video that is no longer on YouTube is reported and skipped.
"""

import argparse
import bisect
import itertools
import json
import statistics
import subprocess
import sys
import urllib.request
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

REPO = "PBatch23888/yunker-chess"
JPEG_QUALITY = 95
SPLITS = ("train", "validation", "test")
# Bits (of 64) a crop's hash may differ from the dataset's by, from JPEG and decoding noise; the median over a video
# is checked
MAX_HASH_DISTANCE = 8
# The dataset's fields kept in metadata.jsonl, after file_name
METADATA_FIELDS = ("fen", "video_id", "game", "frame", "time", "ply", "since_move", "until_move")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv")

# A row of the dataset, or of the metadata.jsonl written beside the crops
Row = dict[str, Any]
# A region as [x, y, width, height], in pixels
Box = Sequence[int]


def read_rows(source: str | Path | None, split: str) -> list[Row]:
    """The split's rows, from a local copy of the dataset or the Hub"""
    if source is not None:
        text = (Path(source) / "data" / f"{split}.jsonl").read_text()
    else:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{split}.jsonl"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode()
    return parse_jsonl(text)


def parse_jsonl(text: str) -> list[Row]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def download(video_id: str, width: int, height: int, videos_dir: Path) -> Path | None:
    """The video's path, downloading it as the dataset's frames were taken from it: YouTube's H.264 stream of that
    size, else another stream of that height"""
    if existing := [p for p in videos_dir.glob(f"{video_id}.*") if p.suffix in VIDEO_SUFFIXES]:
        return existing[0]
    fmt = f"bv*[height={height}][width={width}][vcodec^=avc1]/bv*[height={height}][width={width}]/bv*[height={height}]"
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", fmt, "-o", f"{video_id}.%(ext)s"]
    # Prints the finished file's path, so no partial or temporary file is taken for it
    cmd += ["--print", "after_move:filepath", "-P", str(videos_dir), "--", video_id]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().splitlines()
    return Path(lines[-1]) if result.returncode == 0 and lines else None


def crop(frame: np.ndarray, box: Box, overlay: Box) -> np.ndarray:
    """The frame's board region, `box` as [x, y, width, height], with any part of the overlay in it blacked out"""
    x0, y0, w, h = box
    x1, y1 = x0 + w, y0 + h
    out = frame[y0:y1, x0:x1].copy()
    ox, oy, ow, oh = overlay
    ix0, iy0, ix1, iy1 = max(ox, x0), max(oy, y0), min(ox + ow, x1), min(oy + oh, y1)
    if ix0 < ix1 and iy0 < iy1:
        out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 0
    return out


def dhash(image: np.ndarray) -> str:
    """A 64-bit difference hash of a BGR image, as 16 hex digits: whether each of 8 x 8 grey cells is brighter than the
    one to its left"""
    grey = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA).astype(int)
    bits = (grey[:, 1:] > grey[:, :-1]).flatten()
    return f"{sum(1 << i for i, b in enumerate(bits) if b):016x}"


def hash_distance(a: str, b: str) -> int:
    """Bits two dhash values differ by"""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def scaled(box: Box, sx: float, sy: float) -> list[int]:
    """The box in a video `sx` and `sy` times the size the dataset's was"""
    x, y, w, h = box
    return [round(x * sx), round(y * sy), round(w * sx), round(h * sy)]


def read_frames(
    container: av.container.InputContainer, indices: Sequence[int], fps: float
) -> Iterator[tuple[int, np.ndarray]]:
    """Each of the sorted frame indices found in the video, with its BGR frame; frames are numbered from the first
    at `fps`. Decodes on from the keyframe before each frame wanted, seeking over the stretches between, and knows
    frames by their timestamps, so each is the frame reading them all in order would give"""
    stream = container.streams.video[0]
    start, tick = stream.start_time or 0, float(stream.time_base)

    def index_of(pts: int) -> int:
        return round((pts - start) * tick * fps)

    # Demuxing reads the packets without decoding them, so is quick
    keyframes = sorted(p.pts for p in container.demux(stream) if p.is_keyframe and p.pts is not None)
    key_indices = [index_of(pts) for pts in keyframes]

    def keyframe_before(index: int) -> int:
        return max(bisect.bisect_right(key_indices, index) - 1, 0)

    if not keyframes:
        return
    # One seek for each keyframe that has frames wanted after it, decoding on until they have all been found
    for k, group in itertools.groupby(indices, key=keyframe_before):
        pending = deque(group)
        container.seek(keyframes[k], stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded = index_of(frame.pts)
            while pending and pending[0] < decoded:
                pending.popleft()  # no frame has this index
            if pending and pending[0] == decoded:
                # With BT.601's colours whatever the video's tag says, as OpenCV converts frames, so the crops are
                # those the dataset's hashes were made from
                yield pending.popleft(), frame.reformat(format="bgr24", src_colorspace="ITU601").to_ndarray()
            if not pending:
                break


def crop_video(path: Path, rows: list[Row], out_dir: Path) -> list[Row]:
    """Writes the rows' crops from the video; returns their metadata"""
    metadata: list[Row] = []
    distances: list[int] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)
        width, height = stream.codec_context.width, stream.codec_context.height
        first = rows[0]
        # A download of another size or frame rate than the dataset's: its boxes are scaled, and frames found by time
        sx, sy = width / first["video_width"], height / first["video_height"]
        same_fps = abs(fps - first["fps"]) < 1e-2
        if (sx, sy) != (1, 1) or not same_fps:
            print(
                f"  {path.name} is {width}x{height} at {fps:.3f} fps, not {first['video_width']}x"
                f"{first['video_height']} at {first['fps']:.3f}: the crops may differ a little from the dataset's"
            )
        wanted: defaultdict[int, list[Row]] = defaultdict(list)
        for r in rows:
            wanted[r["frame"] if same_fps else round(r["time"] * fps)].append(r)
        # Seeking by timestamp lands on the exact frame, where OpenCV's seeking by frame number can be a frame or two
        # off
        for index, frame in read_frames(container, sorted(wanted), fps):
            for r in wanted[index]:
                name = f"{r['id']}.jpg"
                image = crop(frame, scaled(r["board_box"], sx, sy), scaled(r["overlay_box"], sx, sy))
                cv2.imwrite(str(out_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                distances.append(hash_distance(dhash(image), r["crop_hash"]))
                metadata.append({"file_name": name, **{k: r[k] for k in METADATA_FIELDS}})
    if missing := len(rows) - len(metadata):
        print(f"  {missing} frames were not found in {path.name} (past its end, or where it skips frames)")
    if distances and (median := statistics.median_low(distances)) > MAX_HASH_DISTANCE:
        print(
            f"  WARNING: {path.name}'s crops differ from the dataset's (median hash distance {median} of 64 bits): "
            "the download may not be the encoding the dataset was made from, so the labels may not fit the crops"
        )
    return metadata


def main(
    out: Path,
    splits: Sequence[str] = SPLITS,
    rows_source: str | Path | None = None,
    videos: Sequence[str] | None = None,
    videos_dir: Path | None = None,
    keep_videos: bool = False,
) -> None:
    """Crops the splits' frames into out (see the module docstring); `videos` limits it to those YouTube ids"""
    videos_dir = videos_dir or out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for split in splits:
        out_dir = out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        by_video: defaultdict[str, list[Row]] = defaultdict(list)
        for r in read_rows(rows_source, split):
            if not videos or r["video_id"] in videos:
                by_video[r["video_id"]].append(r)
        metadata_path = out_dir / "metadata.jsonl"
        done = {m["video_id"] for m in parse_jsonl(metadata_path.read_text())} if metadata_path.is_file() else set()
        for n, (video_id, rows) in enumerate(sorted(by_video.items()), 1):
            if video_id in done:
                continue
            print(f"[{split} {n}/{len(by_video)}] {video_id}: {len(rows)} frames")
            path = download(video_id, rows[0]["video_width"], rows[0]["video_height"], videos_dir)
            if path is None:
                print(f"  could not download {video_id}; skipped")
                failed.append(video_id)
                continue
            metadata = crop_video(path, rows, out_dir)
            # Written per video, so a stopped run starts again where it was
            with metadata_path.open("a") as f:
                f.writelines(json.dumps(m) + "\n" for m in metadata)
            if not keep_videos:
                path.unlink()
    if failed:
        print(f"{len(failed)} videos could not be downloaded: {' '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, action="append", help="default: all three")
    parser.add_argument("--rows", help=f"a local copy of the dataset (default: {REPO} on the Hub)")
    parser.add_argument(
        "--video", action="append", help="only this YouTube id (repeatable; --video=ID for an id starting with -)"
    )
    parser.add_argument("--videos-dir", type=Path, help="where videos are downloaded (default: <out>/videos)")
    parser.add_argument("--keep-videos", action="store_true", help="keep each video once cropped")
    args = parser.parse_args()
    main(args.out, args.split or SPLITS, args.rows, args.video, args.videos_dir, args.keep_videos)
