"""Samryan dataset: 500 overhead DSLR photos of one plastic set on a green roll-up board on a wooden floor, from Sam Ryan
and Mukund Vaidya's chess-dataset (MIT), https://github.com/samryan18/chess-dataset.

Photos are named after their piece placement, ranks joined by ``-``, with ``_2``, ``_3``... on retakes of a position
(``rnbqkbnr-pppppppp-8-8-4P3-8-PPPP1PPP-RNBQKBNR_2.JPG``). White is always at the bottom. Each photo also comes as a
square warp of the playing area (``labeled_preprocessed/<name>.png``) but with no corner labels, so the corners are
found by matching the warp back to the photo (SIFT and a RANSAC homography); the warp's corners land on the board's.

There is one board, one set and one camera, almost straight overhead: little variety, like the Digitizer games. Many
photos are consecutive positions of a few games, so splits are whole groups of positions that link up a few squares
at a time, as in Fenit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote

import chess
import cv2
import numpy as np
import requests
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.chessqueries.data.fenit import assign_splits, games
from camera_chess.config import get_config

REPO = "samryan18/chess-dataset"
COMMIT = "62a8f5b0a45a931c91fd11bf6e1fe0c9fa0d25d9"
PHOTOS_DIR = "labeled_originals"
WARPS_DIR = "labeled_preprocessed"
NAME = re.compile(r"([1-8pnbrqkPNBRQK]+(?:-[1-8pnbrqkPNBRQK]+){7})(?:_\d+)?")
# Matching runs on the photo and its warp shrunk to about this longer side
MATCH_SIDE = 1500
# Near-empty boards give 25-40; valid_corners rejects a bad fit's shape
MIN_INLIERS = 20
# A board's opposite sides, as found, may differ by at most this share (the camera is almost overhead)
MAX_SIDE_MISMATCH = 0.15
# The photos are 6000x4000; crops are shrunk to this shorter side (pack_datasets shrinks further)
MAX_SIDE = 1024
JPEG_QUALITY = 95
WORKERS = 8


def placement_from_name(name: str) -> str | None:
    """A photo's file name -> the piece placement it names, or None when it names none"""
    m = NAME.fullmatch(Path(name).stem)
    if m is None:
        return None
    placement = m.group(1).replace("-", "/")
    try:
        chess.Board(f"{placement} w - - 0 1")
    except ValueError:  # a rank that isn't eight squares
        return None
    return placement


def _upright_size(im: Image.Image) -> tuple[int, int]:
    """An image's (width, height) once turned upright by its EXIF orientation"""
    width, height = im.size
    return (height, width) if im.getexif().get(0x0112, 1) in (5, 6, 7, 8) else (width, height)


def _gray(path: Path, side: int) -> tuple[np.ndarray, float]:
    """An image, upright and grayscale, shrunk to at most ``side`` on its longer side, and the scale from its full
    size"""
    with Image.open(path) as im:
        full_width, _ = _upright_size(im)
        im.draft("L", (side, side))  # a fast JPEG decode at a fraction of full size
        im = ImageOps.exif_transpose(im).convert("L")
    scale = side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    return np.asarray(im), im.width / full_width


def valid_corners(corners: np.ndarray, width: int, height: int) -> bool:
    """Whether TL, TR, BL, BR corners in pixels make a convex board inside the photo with near-equal opposite sides"""
    tl, tr, bl, br = corners
    if not cv2.isContourConvex(np.array([tl, tr, br, bl], dtype=np.float32).reshape(-1, 1, 2)):
        return False
    if (corners < 0).any() or (corners[:, 0] > width).any() or (corners[:, 1] > height).any():
        return False
    pairs = [(tr - tl, br - bl), (bl - tl, br - tr)]
    return all(
        abs(np.linalg.norm(a) - np.linalg.norm(b)) <= MAX_SIDE_MISMATCH * max(np.linalg.norm(a), np.linalg.norm(b))
        for a, b in pairs
    )


def board_corners(photo: Path, warp: Path) -> list[tuple[float, float]] | None:
    """The playing area's TL, TR, BL, BR corners in the (EXIF-upright) photo's pixels, or None when the warp can't be
    matched back to it"""
    with Image.open(photo) as im:
        width, height = _upright_size(im)
    photo_px, photo_scale = _gray(photo, MATCH_SIDE)
    warp_px, _warp_scale = _gray(warp, MATCH_SIDE * 2 // 3)
    sift = cv2.SIFT_create()  # ty: ignore[unresolved-attribute]  # missing from the stubs
    warp_kp, warp_desc = sift.detectAndCompute(warp_px, None)
    photo_kp, photo_desc = sift.detectAndCompute(photo_px, None)
    if warp_desc is None or photo_desc is None or len(photo_kp) < 2:
        return None
    good = [m for m, n in cv2.BFMatcher().knnMatch(warp_desc, photo_desc, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.array([warp_kp[m.queryIdx].pt for m in good], dtype=np.float32)
    dst = np.array([photo_kp[m.trainIdx].pt for m in good], dtype=np.float32)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None or mask.sum() < MIN_INLIERS:
        return None
    h, w = warp_px.shape
    corners = cv2.perspectiveTransform(
        np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32).reshape(-1, 1, 2), homography
    )
    corners = corners.reshape(-1, 2) / photo_scale
    if not valid_corners(corners, width, height):
        return None
    return [(float(x), float(y)) for x, y in corners]


def crop_box(corners: list[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the corners' bounding box, kept within the photo"""
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (
        max(0, round(min(xs))),
        max(0, round(min(ys))),
        min(width, round(max(xs))),
        min(height, round(max(ys))),
    )


def git_blob_sha(data: bytes) -> str:
    """The SHA-1 git names a file's contents by, as the tree listing gives it"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _fetch(session: requests.Session, path: str, sha: str, dest: Path) -> Path:
    """The repo file at ``path``, downloaded once to ``dest`` and checked against its git blob SHA"""
    if dest.is_file() and git_blob_sha(dest.read_bytes()) == sha:
        return dest
    r = session.get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{quote(path)}", timeout=120)
    r.raise_for_status()
    if git_blob_sha(r.content) != sha:
        raise RuntimeError(f"SHA mismatch for {path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(r.content)
    part.replace(dest)
    return dest


def download(root: Path) -> list[tuple[Path, Path]]:
    """Fetch the photos and their warps at ``COMMIT`` to ``root/source`` -> [(photo, its warp)], sorted by name"""
    session = requests.Session()
    r = session.get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", timeout=60)
    r.raise_for_status()
    tree = r.json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub truncated the tree")
    blobs = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    warps = {Path(p).stem: p for p in blobs if p.startswith(f"{WARPS_DIR}/")}
    photos = sorted(p for p in blobs if p.startswith(f"{PHOTOS_DIR}/"))
    unmatched = [p for p in photos if Path(p).stem not in warps]
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} photo(s) without a warp, e.g. {unmatched[:3]}")
    pairs = [(p, warps[Path(p).stem]) for p in photos]
    jobs = [(path, blobs[path], root / "source" / path) for pair in pairs for path in pair]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(tqdm(pool.map(lambda job: _fetch(session, *job), jobs), total=len(jobs), desc="samryan download"))
    return [(root / "source" / photo, root / "source" / warp) for photo, warp in pairs]


def build(root: Path | None = None) -> Path:
    """Download the dataset to ``root/source`` and write ``root/images/<split>/*.jpg`` and ``root/annotations.json``.
    Downloads are kept, so a rerun only fetches what is missing."""
    root = Path(root) if root else get_config().samryan_root
    pairs = download(root)
    with ThreadPoolExecutor(WORKERS) as pool:
        found = list(tqdm(pool.map(lambda pair: board_corners(*pair), pairs), total=len(pairs), desc="samryan corners"))
    photos = []  # (photo, its placement, its board corners)
    skipped = defaultdict(list)
    for (photo, _), corners in zip(pairs, found, strict=True):
        placement = placement_from_name(photo.name)
        if placement is None or corners is None:
            skipped["no placement in the name" if placement is None else "no board match"].append(photo.name)
            continue
        photos.append((photo, placement, corners))
    for reason, names in skipped.items():
        logger.info(f"  [skip] {len(names)} photo(s) with {reason}: {names}")

    # One camera, so every photo gets the same size key
    groups = games([(p.name, placement, (0, 0)) for p, placement, _ in photos])
    group_of = {i: n for n, group in enumerate(sorted(groups, key=min)) for i in group}
    splits = assign_splits(groups)
    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        (root / "images" / split.value).mkdir(parents=True, exist_ok=True)

    def write_crop(i: int) -> dict:
        path, placement, corners = photos[i]
        image = f"images/{splits[i].value}/samryan_{i:03d}.jpg"
        with Image.open(path) as im:
            photo = ImageOps.exif_transpose(im).convert("RGB")
        box = crop_box(corners, photo.width, photo.height)
        crop = photo.crop(box)
        scale = min(1.0, MAX_SIDE / min(crop.size))
        if scale < 1:
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        crop.save(root / image, format="JPEG", quality=JPEG_QUALITY)
        return {
            "image": image,
            "fen": placement,
            "split": splits[i].value,
            "group": f"samryan_g{group_of[i]:03d}",
            "source": path.name,
            # In the upright source photo's pixels; the crop is box, shrunk by scale
            "corners": [[round(x, 1), round(y, 1)] for x, y in corners],
            "box": list(box),
            "scale": round(scale, 6),
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        annotations = list(tqdm(pool.map(write_crop, range(len(photos))), total=len(photos), desc="samryan crops"))
    (root / "annotations.json").write_text(json.dumps(annotations, indent=1))
    logger.info(f"  [ok] {len(annotations)} samryan crops in {len(groups)} games in {root}")
    return root


class Samryan(ChessDataset):
    name = DatasetName.SAMRYAN
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 408, Split.VAL: 46, Split.TEST: 46})

    @staticmethod
    def default_root() -> Path:
        return get_config().samryan_root

    def _annotations(self) -> list[dict]:
        path = self.root / "annotations.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; build it with camera_chess.chessqueries.data.samryan.build")
        return json.loads(path.read_text())

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        assert split is not None  # _validate_split requires one
        return [
            BoardSample(
                image_path=self.root / a["image"],
                board=Board.from_fen(a["fen"]),
                dataset=self.name,
                sample_id=Path(a["image"]).stem,
                split=split,
                # The board corners are in the source photo; the crop is its box, shrunk by scale
                meta={"group": a["group"], "source": a["source"], "corners": a["corners"], "box": a["box"]},
            )
            for a in self._annotations()
            if a["split"] == split.value
        ]
