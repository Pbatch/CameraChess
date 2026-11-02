"""Find the 49 x-corners automatically for video frames that have no hand labels (e.g. nyh).

The model rarely finds all 49 corners in every frame, but the camera is static within a video,
so frames where it does (and they fit a clean 7x7 grid) lend their corners to the rest of the
video, mapped across by aligning the frames to follow any camera drift.

    uv run python -m camera_chess.labels.auto_xcorners                # write xcorners_auto.json for nyh
    uv run python -m camera_chess.labels.auto_xcorners --validate     # compare with hand labels on other videos
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import orjson
from tqdm import tqdm

from camera_chess.constants import LABELS_DIR
from camera_chess.labels.make_chessqueries_dataset import (
    PHOTO_PREFIXES,
    _estimate_alignment,
    fit_lattice,
    hand_xcorners,
    is_excluded,
    source,
)
from camera_chess.labels.xcorner_model import XcornerDetector

ROOT = Path(LABELS_DIR)
OUT = ROOT / "xcorners_auto.json"


def video_frames(split: str, want) -> dict[str, list[str]]:
    videos = defaultdict(list)
    for p in (ROOT / "pieces" / split / "images").glob("*.jpg"):
        if want(p.stem):
            videos[source(p.stem)].append(p.stem)
    for names in videos.values():
        names.sort(key=lambda n: int(n.rsplit("_", 1)[1]))
    return videos


def image_path(split: str, name: str) -> Path:
    return ROOT / "pieces" / split / "images" / f"{name}.jpg"


def fit_frames(detector: XcornerDetector, split: str, names: list[str]) -> dict[str, np.ndarray]:
    """The frames whose detections alone fit a clean 7x7 grid (GPU pass)"""
    fitted = {}
    for name in names:
        points, _ = detector(cv2.cvtColor(cv2.imread(str(image_path(split, name))), cv2.COLOR_BGR2RGB))
        # Extra detections are usually spurious, so keep the 49 most confident
        if len(points) >= 49 and fit_lattice(points[:49]) is not None:
            fitted[name] = points[:49]
    return fitted


def spread_to_video(split: str, names: list[str], fitted: dict[str, np.ndarray]) -> dict[str, list]:
    """49 x-corners for every frame the video's fitted frames can reach (CPU pass, one video)"""
    cv2.setNumThreads(1)  # parallel across videos instead
    if not fitted:
        return {}
    order = {n: i for i, n in enumerate(names)}
    result = {n: p.tolist() for n, p in fitted.items()}
    for name in names:
        if name in fitted:
            continue
        # Borrow from the nearest frame that has its own fit, mapped across the camera's drift
        ref = min(fitted, key=lambda n: abs(order[n] - order[name]))
        M = _estimate_alignment(image_path(split, ref), image_path(split, name))
        if M is not None:
            M = np.array(M)
            result[name] = (fitted[ref] @ M[:, :2].T + M[:, 2]).tolist()
    return result


def _spread_job(job: tuple) -> dict[str, list]:
    return spread_to_video(*job)


def find_xcorners(detector: XcornerDetector, videos: list[tuple[str, list[str]]]) -> list[dict[str, list]]:
    """`spread_to_video` for each (split, names) video, detecting on the GPU then spreading on all cores"""
    jobs = [(split, names, fit_frames(detector, split, names)) for split, names in tqdm(videos, desc="detect")]
    # Spawned, not forked: forking after OpenCV and CUDA have started threads deadlocks the workers
    with get_context("spawn").Pool() as pool:
        return list(tqdm(pool.imap(_spread_job, jobs), total=len(jobs), desc="spread"))


def board_corner_error(auto: np.ndarray, hand: np.ndarray) -> float | None:
    """Largest gap between the two grids' outer board corners, in squares (orientation-free)"""
    Ha, Hh = fit_lattice(auto), fit_lattice(hand)
    if Ha is None or Hh is None:
        return None
    outer = np.array([[[-1, -1], [7, -1], [7, 7], [-1, 7]]], dtype=np.float32)
    pa = cv2.perspectiveTransform(outer, np.linalg.inv(Ha))[0]
    ph = cv2.perspectiveTransform(outer, np.linalg.inv(Hh))[0]
    square = np.linalg.norm(ph[1] - ph[0]) / 8
    # The two fits may number the corners differently, so match each corner to its nearest
    return float(max(np.linalg.norm(ph - p, axis=1).min() for p in pa) / square)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    detector = XcornerDetector()

    if args.validate:
        videos = [
            (split, names)
            for split in ("train", "val")
            for names in video_frames(
                split,
                lambda n, split=split: (
                    not n.startswith(("nyh",) + PHOTO_PREFIXES)
                    and not is_excluded(n)
                    and (ROOT / "xcorners" / split / "labels" / f"{n}.txt").exists()
                ),
            ).values()
        ]
        errors, covered = [], 0
        for (split, names), found in zip(videos, find_xcorners(detector, videos)):
            covered += len(found)
            for name, pts in found.items():
                hand = hand_xcorners(name, split, ROOT)
                if hand is not None and (e := board_corner_error(np.array(pts), hand)) is not None:
                    errors.append(e)
        total = sum(len(names) for _, names in videos)
        errors = np.array(errors)
        print(
            f"frames covered {covered}/{total}; board corner error (squares): median {np.median(errors):.3f}, "
            f"95th pct {np.percentile(errors, 95):.3f}; > 0.25 square: {np.mean(errors > 0.25):.1%}, "
            f"> 0.5: {np.mean(errors > 0.5):.1%}"
        )
        return

    videos = [
        (split, names)
        for split in ("train", "val")
        for names in video_frames(split, lambda n: n.startswith("nyh")).values()
    ]
    result = {}
    for (split, _), found in zip(videos, find_xcorners(detector, videos)):
        result.update({f"{split}/{name}": pts for name, pts in found.items()})
    OUT.write_bytes(orjson.dumps(result))
    print(f"{len(result)} frames with x-corners in {OUT}")


if __name__ == "__main__":
    main()
