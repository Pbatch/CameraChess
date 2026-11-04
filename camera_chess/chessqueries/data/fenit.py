"""Fenit dataset: phone photos of chess sets on folding boards on a tiled floor, from "Annotated Image Dataset for
Fenit: Automatic Chessboard Digitization and FEN Generation" (Harjanto, Oei and Susanto, Binus University; CC BY 4.0),
https://data.mendeley.com/datasets/wpp9xnbcp6/1 (doi:10.17632/wpp9xnbcp6.1).

Most photos are named after their piece placement, ranks joined by ``_`` or ``-``, sometimes followed by the side to
move or a copy number (``8_8_1p1k4_5ppp_PPK1p3_6P1_5PP1_8 b (2).jpg``). Each comes with a YOLO-pose label of the same
name: one row per object, ``class cx cy w h`` then four ``x y visibility`` keypoints, normalised to the photo. Class 0
is the board, whose keypoints are its four corners.

The placement is taken from the name, and only the board corners from the label: the labels' piece classes come from
a detector and are sometimes wrong (a black king boxed as a pawn, a row of white pawns as black), and the corners are
sometimes on the board's frame rather than the playing area, which is close enough for a crop. Photos are skipped when
their name holds no valid placement (close-ups of single pieces, timestamped photos), their label has no board, or two
photos share one label.

Many photos are consecutive positions of one game in one session, so splits are whole groups of photos taken with the
same camera whose positions link up a few squares at a time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType

import chess
import requests
from loguru import logger
from PIL import Image
from tqdm import tqdm

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config

DATASET_ID = "wpp9xnbcp6"
VERSION = 1
# The dataset record, with every file's download link and SHA-256 (the file-listing API stops at 100 files)
DATASET_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
IMAGE_SUFFIXES = (".jpg", ".jpeg")
NAME = re.compile(r"([1-8pnbrqkPNBRQK]+(?:[_-][1-8pnbrqkPNBRQK]+){7})(?: [wb])?\s*(?:\(.*\))?")
# Positions this many squares apart, photographed with the same camera, are taken to be from one game
SAME_GAME_SQUARES = 4
# Shares of the photos in each split, of whole games
SPLIT_SHARES = {Split.TRAIN: 0.7, Split.VAL: 0.15, Split.TEST: 0.15}
# Games with more photos than this are all for training, so no one game makes up most of val or test
MAX_HELD_OUT_GAME = 10
# Share of the corners' bounding box added on every side of a crop, as in camera_chess/videos/make_training_data.py
MARGIN = 0.15
# The photos are 3000x4000 to 4284x5712; crops are shrunk to this shorter side (pack_datasets shrinks further)
MAX_SIDE = 1024
JPEG_QUALITY = 95
DOWNLOAD_WORKERS = 8


def placement_from_name(name: str) -> str | None:
    """A photo's file name -> the piece placement it names, or None when it names none"""
    m = NAME.fullmatch(Path(name).stem)
    if m is None:
        return None
    placement = re.sub("[_-]", "/", m.group(1))
    try:
        chess.Board(f"{placement} w - - 0 1")
    except ValueError:  # a rank that isn't eight squares
        return None
    return placement


def board_corners(label: str) -> list[tuple[float, float]] | None:
    """A YOLO-pose label -> its board's four corners, normalised, or None when it has no board"""
    corners = None
    for line in label.splitlines():
        values = line.split()
        if values and values[0] == "0":
            if len(values) != 17:
                raise ValueError(f"expected 17 values in a board row, got {len(values)}: {line!r}")
            if corners is not None:
                raise ValueError("more than one board")
            corners = [(float(values[i]), float(values[i + 1])) for i in range(5, 17, 3)]
    return corners


def crop_box(corners: list[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) in pixels: the corners' bounding box, grown by ``MARGIN`` of its size and kept within the photo"""
    xs, ys = [x * width for x, _ in corners], [y * height for _, y in corners]
    dx, dy = MARGIN * (max(xs) - min(xs)), MARGIN * (max(ys) - min(ys))
    return (
        max(0, round(min(xs) - dx)),
        max(0, round(min(ys) - dy)),
        min(width, round(max(xs) + dx)),
        min(height, round(max(ys) + dy)),
    )


def _squares_apart(a: str, b: str) -> int:
    pa, pb = chess.Board(f"{a} w - - 0 1").piece_map(), chess.Board(f"{b} w - - 0 1").piece_map()
    return sum(pa.get(s) != pb.get(s) for s in set(pa) | set(pb))


def games(photos: list[tuple[str, str, tuple[int, int]]]) -> list[list[int]]:
    """(name, placement, photo size) per photo -> groups of their indices, each a game: photos with the same size
    whose placements link up at most ``SAME_GAME_SQUARES`` squares at a time"""
    parent = list(range(len(photos)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, placement, size) in enumerate(photos):
        for j in range(i):
            if photos[j][2] == size and _squares_apart(placement, photos[j][1]) <= SAME_GAME_SQUARES:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(photos)):
        groups[find(i)].append(i)
    return list(groups.values())


def assign_splits(groups: list[list[int]]) -> dict[int, Split]:
    """Photo index -> split, for whole groups: those over ``MAX_HELD_OUT_GAME`` photos to train, then the rest in
    order of their first photo, each to the split furthest below its share"""
    total = sum(len(g) for g in groups)
    counts = dict.fromkeys(SPLIT_SHARES, 0)
    out = {}
    for group in sorted(groups, key=lambda g: (len(g) <= MAX_HELD_OUT_GAME, min(g))):
        if len(group) > MAX_HELD_OUT_GAME:
            split = Split.TRAIN
        else:
            split = min(SPLIT_SHARES, key=lambda s: (counts[s] - SPLIT_SHARES[s] * total) / SPLIT_SHARES[s])
        counts[split] += len(group)
        out.update(dict.fromkeys(group, split))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(session: requests.Session, entry: dict, dest: Path) -> Path:
    """One listed file, downloaded once to ``dest`` and checked against the listing's SHA-256"""
    expected = entry["content_details"]["sha256_hash"]
    if dest.is_file() and _sha256(dest) == expected:
        return dest
    r = session.get(entry["content_details"]["download_url"], timeout=120)
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest() != expected:
        raise RuntimeError(f"SHA-256 mismatch for {entry['filename']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(r.content)
    part.replace(dest)
    return dest


def download(root: Path) -> list[tuple[Path, Path]]:
    """Fetch version ``VERSION`` to ``root/source/{images,labels}`` -> [(photo, its label)], sorted by name.

    Photos without a label, and photos sharing a label (``x.jpg`` and ``x.jpeg``), are left out."""
    session = requests.Session()
    r = session.get(DATASET_URL, timeout=60)
    r.raise_for_status()
    record = r.json()
    if record["version"] != VERSION:
        raise RuntimeError(f"{DATASET_URL} is version {record['version']}, expected {VERSION}")
    by_stem: dict[str, list[dict]] = defaultdict(list)
    labels = {}
    for entry in record["files"]:
        path = Path(entry["filename"])
        if path.suffix.lower() in IMAGE_SUFFIXES:
            by_stem[path.stem].append(entry)
        elif path.suffix == ".txt":
            labels[path.stem] = entry
    shared = sorted(stem for stem, entries in by_stem.items() if len(entries) > 1)
    if shared:
        logger.info(f"  [skip] {len(shared)} label(s) shared by two photos: {shared}")
    pairs = [
        (entries[0], labels[stem]) for stem, entries in sorted(by_stem.items()) if len(entries) == 1 and stem in labels
    ]
    jobs = [(image, root / "source" / "images" / image["filename"]) for image, _ in pairs]
    jobs += [(label, root / "source" / "labels" / label["filename"]) for _, label in pairs]
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        list(tqdm(pool.map(lambda job: _fetch(session, *job), jobs), total=len(jobs), desc="fenit download"))
    return [
        (root / "source" / "images" / image["filename"], root / "source" / "labels" / label["filename"])
        for image, label in pairs
    ]


def build(root: Path | None = None) -> Path:
    """Download the dataset to ``root/source`` and write ``root/images/<split>/*.jpg`` and ``root/annotations.json``.
    Downloads are kept, so a rerun only fetches what is missing."""
    root = Path(root) if root else get_config().fenit_root
    photos = []  # (photo, its placement, its size, its board corners)
    skipped = defaultdict(int)
    for image, label in download(root):
        placement = placement_from_name(image.name)
        corners = board_corners(label.read_text())
        if placement is None or corners is None:
            skipped["no placement in the name" if placement is None else "no board in the label"] += 1
            continue
        with Image.open(image) as im:
            photos.append((image, placement, im.size, corners))
    for reason, count in skipped.items():
        logger.info(f"  [skip] {count} photo(s) with {reason}")

    groups = games([(p.name, placement, size) for p, placement, size, _ in photos])
    group_of = {i: n for n, group in enumerate(sorted(groups, key=min)) for i in group}
    splits = assign_splits(groups)
    for split in SPLIT_SHARES:
        (root / "images" / split.value).mkdir(parents=True, exist_ok=True)
    annotations = []
    for i, (path, placement, (width, height), corners) in enumerate(tqdm(photos, desc="fenit crops")):
        box = crop_box(corners, width, height)
        image = f"images/{splits[i].value}/fenit_{i:03d}.jpg"
        with Image.open(path) as photo:
            crop = photo.convert("RGB").crop(box)
        scale = min(1.0, MAX_SIDE / min(crop.size))
        if scale < 1:
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        crop.save(root / image, format="JPEG", quality=JPEG_QUALITY)
        annotations.append(
            {
                "image": image,
                "fen": placement,
                "split": splits[i].value,
                "group": f"fenit_g{group_of[i]:03d}",
                "source": path.name,
                # In the source photo's pixels; the crop is box, shrunk by scale
                "corners": [[round(x * width, 1), round(y * height, 1)] for x, y in corners],
                "box": list(box),
                "scale": round(scale, 6),
            }
        )
    (root / "annotations.json").write_text(json.dumps(annotations, indent=1))
    logger.info(f"  [ok] {len(annotations)} fenit crops in {len(groups)} games in {root}")
    return root


class Fenit(ChessDataset):
    name = DatasetName.FENIT
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 268, Split.VAL: 58, Split.TEST: 58})

    @staticmethod
    def default_root() -> Path:
        return get_config().fenit_root

    def _annotations(self) -> list[dict]:
        path = self.root / "annotations.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; build it with camera_chess.chessqueries.data.fenit.build")
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
