"""Digitizer dataset: webcam frames of 20 games from Oliver Butler's Realtime-Chess-Digitizer (CC0), also on Kaggle as
oliverbutlersu/chess-pieces-and-board-recognition.

Each game folder holds one frame per position (the start, then one after every ply) and a ``moves.txt``:

    <rotation>                      quarter turns of the board in the frame, 0-3
    <chess.com link>
    <x y x y x y x y>               board corners: top-left, top-right, bottom-left, bottom-right
    1. e4 e5 ...                    the game, one move pair per line

so frame ``i``, in filename (timestamp) order, shows the position after ``i`` plies. ``build`` replays each game and
writes board crops plus ``annotations.json``. There is one board, one set and one camera, almost straight overhead:
little variety, so splits are whole games.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote

import chess
import requests
from loguru import logger
from PIL import Image
from tqdm import tqdm

from camera_chess.chessqueries.core import Board, Split
from camera_chess.chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from camera_chess.config import get_config

REPO = "OliverButlerSU/Realtime-Chess-Digitizer"
COMMIT = "9fdee85ffeab071b3e871f90a6b3f6aa3c4f74bb"
GAMES_DIR = "Test Images/Chess Games"
GAMES = range(1, 21)
# Games 15 and 20 are the same game recorded twice, so they share a split
TEST_GAMES = frozenset({6, 15, 20})
VAL_GAMES = frozenset({3, 17})
# Of the board's side, around the corners; the camera is above the centre, so pieces lean out past the edges
MARGIN = 0.1
JPEG_QUALITY = 95
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def game_split(game: int) -> Split:
    if game in TEST_GAMES:
        return Split.TEST
    if game in VAL_GAMES:
        return Split.VAL
    return Split.TRAIN


def parse_moves(text: str) -> tuple[int, list[tuple[int, int]], list[str]]:
    """``moves.txt`` -> (rotation, corners as TL, TR, BL, BR, SAN moves)"""
    lines = text.splitlines()
    rotation = int(lines[0])
    numbers = [int(n) for n in lines[2].split()]
    if len(numbers) != 8:
        raise ValueError(f"expected 8 corner coordinates, got {lines[2]!r}")
    corners = list(zip(numbers[::2], numbers[1::2], strict=True))
    tokens = re.sub(r"\d+\.", " ", " ".join(lines[3:])).split()
    return rotation, corners, [t for t in tokens if t not in RESULTS]


def placements(moves: list[str]) -> list[str]:
    """The piece placement before the first move and after every ply"""
    board = chess.Board()
    out = [board.board_fen()]
    for san in moves:
        board.push_san(san)
        out.append(board.board_fen())
    return out


def crop_box(corners: list[tuple[int, int]], width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the corners' bounding box, grown by ``MARGIN`` of its size and kept within the frame"""
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    dx, dy = MARGIN * (max(xs) - min(xs)), MARGIN * (max(ys) - min(ys))
    return (
        max(0, round(min(xs) - dx)),
        max(0, round(min(ys) - dy)),
        min(width, round(max(xs) + dx)),
        min(height, round(max(ys) + dy)),
    )


def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{quote(path)}"


def _game_files(session: requests.Session) -> dict[int, list[str]]:
    """Game -> its frame paths in the repo, in timestamp order"""
    # The whole repo is too big to list at once (GitHub truncates it), so walk down to the games first
    sha = COMMIT
    for name in GAMES_DIR.split("/"):
        r = session.get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}", timeout=60)
        r.raise_for_status()
        sha = next(e["sha"] for e in r.json()["tree"] if e["path"] == name)
    r = session.get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}?recursive=1", timeout=60)
    r.raise_for_status()
    tree = r.json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub truncated the games tree")
    frames: dict[int, list[str]] = {g: [] for g in GAMES}
    for entry in tree["tree"]:
        m = re.fullmatch(r"Game (\d+)/[^/]+\.jpg", entry["path"])
        if entry["type"] == "blob" and m:
            frames[int(m.group(1))].append(f"{GAMES_DIR}/{entry['path']}")
    return {g: sorted(paths) for g, paths in frames.items()}


def _fetch(session: requests.Session, path: str, dest: Path) -> Path:
    """The repo file at ``path``, downloaded once to ``dest``"""
    if not dest.is_file():
        r = session.get(_raw_url(path), timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(r.content)
        part.replace(dest)
    return dest


def build(root: Path | None = None) -> Path:
    """Download the games at ``COMMIT`` to ``root/source`` and write ``root/images/<split>/*.jpg`` and
    ``root/annotations.json``. Downloads are kept, so a rerun only fetches what is missing."""
    root = Path(root) if root else get_config().digitizer_root
    session = requests.Session()
    frames = _game_files(session)
    annotations = []
    for game in tqdm(GAMES, desc="digitizer games"):
        moves_path = f"{GAMES_DIR}/Game {game}/moves.txt"
        rotation, corners, moves = parse_moves(_fetch(session, moves_path, root / "source" / moves_path).read_text())
        fens = placements(moves)
        if len(fens) != len(frames[game]):
            raise RuntimeError(f"game {game}: {len(frames[game])} frames for {len(fens)} positions")
        split = game_split(game)
        (root / "images" / split.value).mkdir(parents=True, exist_ok=True)
        for ply, (path, fen) in enumerate(zip(frames[game], fens, strict=True)):
            image = f"images/{split.value}/digitizer_g{game:02d}_{ply:03d}.jpg"
            with Image.open(_fetch(session, path, root / "source" / path)) as frame:
                box = crop_box(corners, frame.width, frame.height)
                frame.convert("RGB").crop(box).save(root / image, format="JPEG", quality=JPEG_QUALITY)
            annotations.append(
                {
                    "image": image,
                    "fen": fen,
                    "split": split.value,
                    "game": game,
                    "ply": ply,
                    "source": path,
                    "rotation": rotation,
                    "corners": corners,
                    "box": list(box),
                }
            )
    (root / "annotations.json").write_text(json.dumps(annotations, indent=1))
    logger.info(f"  [ok] {len(annotations)} digitizer crops in {root}")
    return root


class Digitizer(ChessDataset):
    name = DatasetName.DIGITIZER
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = MappingProxyType({Split.TRAIN: 1027, Split.VAL: 141, Split.TEST: 188})

    @staticmethod
    def default_root() -> Path:
        return get_config().digitizer_root

    def _annotations(self) -> list[dict]:
        path = self.root / "annotations.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; build it with camera_chess.chessqueries.data.digitizer.build")
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
                # The board corners are in the source frame; the crop starts at box[:2]
                meta={"game": a["game"], "ply": a["ply"], "corners": a["corners"], "box": a["box"]},
            )
            for a in self._annotations()
            if a["split"] == split.value
        ]
