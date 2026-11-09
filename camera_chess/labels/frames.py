"""Frames of the YOLO pieces + xcorners labels, placed on the 8x8 grid, shared by the dataset build and the
labelling tools.

A frame's 49 x-corners fit a lattice, and each piece's anchor point is matched to the square under it. The
lattice's rotation (which corner is a1) is not labelled, so squares are only known up to `rotate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
import orjson
from PIL import Image
from scipy.optimize import linear_sum_assignment

CLASSES = [
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]
SYMBOLS = "bknpqrBKNPQR"  # FEN symbol of each class above
# Duplicates ChessReD, so would leak its test set into training
EXCLUDED_PREFIXES = ("chessred2k",)
# Independent photos rather than frames of one video, so each gets its own orientation
PHOTO_PREFIXES = ("roboflow", "google")
# Photo sources whose every photo was skipped when labelling orientation by hand
EXCLUDED_SOURCES = {"roboflow_11", "roboflow_3", "roboflow_4", "roboflow_6", "roboflow_final_chess_piece"}


def source(name: str) -> str:
    """The dataset or video an image came from, e.g. roboflow_11 for roboflow_11_68"""
    return name.rsplit("_", 1)[0]


def is_excluded(name: str) -> bool:
    return name.startswith(EXCLUDED_PREFIXES) or source(name) in EXCLUDED_SOURCES


@dataclass
class Frame:
    name: str
    split: str
    image: Path
    group: str
    # (n, 2) file/rank of each on-board piece in lattice coordinates, and its class
    squares: np.ndarray
    classes: np.ndarray
    off_board: int
    # Homography from the image's pixels to lattice coords
    H: np.ndarray


def read_yolo(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array([line.split() for line in path.read_text().splitlines() if line.strip()], dtype=np.float64)
    if rows.size == 0:
        return np.zeros(0, int), np.zeros((0, 4))
    cls = rows[:, 0].astype(int)
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return cls, np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def fit_lattice(xcorners: np.ndarray) -> np.ndarray | None:
    """Homography from image pixels to lattice coords, where the x-corners sit at integers 0..6.

    The lattice is right-handed like a real board seen from above (files then ranks), but its
    rotation is arbitrary until one is chosen for the frame's group."""
    hull = cv2.convexHull(xcorners.astype(np.float32))
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            quad = approx[:, 0]
            break
    if quad is None:
        return None
    target = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    lattice = cv2.perspectiveTransform(xcorners[None].astype(np.float32), H)[0]
    rounded = np.round(lattice)
    if len({tuple(p) for p in rounded}) != len(xcorners) or np.abs(lattice - rounded).max() > 0.3:
        return None
    H, _ = cv2.findHomography(xcorners.astype(np.float32), rounded.astype(np.float32))

    # Files run left to right and ranks bottom to top when White is at the bottom of the image,
    # which in y-down pixels makes the cross product of the two directions negative
    inv = np.linalg.inv(H)
    o, u, v = cv2.perspectiveTransform(np.array([[[3, 3], [4, 3], [3, 4]]], dtype=np.float32), inv)[0]
    du, dv = u - o, v - o
    if du[0] * dv[1] - du[1] * dv[0] > 0:
        H = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) @ H  # swap axes
    return H


def rotate(squares: np.ndarray, k: int) -> np.ndarray:
    """Rotate (file, rank) squares by k quarter turns about the board centre."""
    f, r = squares[:, 0], squares[:, 1]
    for _ in range(k):
        f, r = r, 7 - f
    return np.stack([f, r], axis=1)


def to_labels(squares: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """64 class indices (-1 for empty) in FEN order, a8..h1"""
    labels = -np.ones(64, dtype=int)
    labels[(7 - squares[:, 1]) * 8 + squares[:, 0]] = classes
    return labels


def agreement(frame: Frame, k: int, prediction: list[int]) -> int:
    """Squares on which the rotated labels match the model's prediction"""
    return int(np.sum(to_labels(rotate(frame.squares, k), frame.classes) == np.array(prediction)))


def choose_rotation_by_model(frames: list[Frame], predictions: dict[str, list[int]]) -> tuple[int, float]:
    """The rotation agreeing most with the model, and its lead over the runner-up in squares per frame."""
    totals = sorted(((sum(agreement(f, k, predictions[f.name]) for f in frames), k) for k in range(4)), reverse=True)
    return totals[0][1], (totals[0][0] - totals[1][0]) / len(frames)


def to_fen(squares: np.ndarray, classes: np.ndarray) -> str:
    board = chess.BaseBoard.empty()
    for (f, r), c in zip(squares, classes):
        board.set_piece_at(chess.square(f, r), chess.Piece.from_symbol(SYMBOLS[c]))
    return board.board_fen()


def apply_label_fixes(fen: str, fixes: dict[str, list[str]]) -> tuple[str, dict[str, list[str]]]:
    """The FEN with each fix ({square: [label, correction]}, '' for empty) applied, and the fixes used. A fix only
    applies while its square still holds the label that was reviewed, so a relabelled square isn't overwritten."""
    board = chess.BaseBoard(fen)
    used = {}
    for square, (label, correction) in fixes.items():
        index = chess.parse_square(square)
        piece = board.piece_at(index)
        if (piece.symbol() if piece else "") == label:
            if correction:
                board.set_piece_at(index, chess.Piece.from_symbol(correction))
            else:
                board.remove_piece_at(index)
            used[square] = [label, correction]
    return board.board_fen(), used


SQUARE_CENTRES = np.array([(f, r) for f in range(8) for r in range(8)], dtype=np.float32) - 0.5


def assign_squares(lattice: np.ndarray, margin: float) -> np.ndarray:
    """Square (file, rank) for each piece anchor in lattice coords, or (-1, -1) if off the board.

    Anchors within `margin` squares outside the board still count as on it, since steep views
    push the anchors of back-rank pieces just past the edge. Pieces are matched one-to-one to the
    nearest free square centres, so two anchors falling in one square don't collide."""
    squares = -np.ones((len(lattice), 2), dtype=int)
    near = np.all((lattice >= -1 - margin) & (lattice <= 7 + margin), axis=1)
    if near.any():
        cost = np.linalg.norm(lattice[near, None] - SQUARE_CENTRES[None], axis=2)
        rows, cols = linear_sum_assignment(cost)
        squares[np.flatnonzero(near)[rows]] = SQUARE_CENTRES[cols] + 0.5
    return squares


ALIGNMENT_CACHE = "xcorner_alignment.json"
_alignments: dict[str, list | None] | None = None
_orb = cv2.ORB.create(3000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _estimate_alignment(xcorners_image: Path, pieces_image: Path) -> list | None:
    """2x3 similarity transform from the xcorners copy of an image to its pieces copy, or None.

    The two copies are different crops (and zooms) of the same frame, so the x-corners only line
    up with the pieces once mapped across. Fits that rotate, match few features, or leave the
    images looking different are rejected."""
    a, b = cv2.imread(str(xcorners_image), cv2.IMREAD_GRAYSCALE), cv2.imread(str(pieces_image), cv2.IMREAD_GRAYSCALE)
    if a.shape == b.shape and np.abs(a.astype(float) - b).mean() < 3:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ka, da = _orb.detectAndCompute(a, None)  # ty: ignore[no-matching-overload]  # stubs miss the optional mask
    kb, db = _orb.detectAndCompute(b, None)  # ty: ignore[no-matching-overload]  # stubs miss the optional mask
    if da is None or db is None:
        return None
    matches = _matcher.match(da, db)
    if len(matches) < 30:
        return None
    pa = np.array([ka[m.queryIdx].pt for m in matches], np.float32)
    pb = np.array([kb[m.trainIdx].pt for m in matches], np.float32)
    M, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2)
    if M is None or inliers.sum() < 50 or abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))) > 1:
        return None
    warped = cv2.warpAffine(a, M, (b.shape[1], b.shape[0]))
    mask = cv2.warpAffine(np.ones_like(a), M, (b.shape[1], b.shape[0])) > 0
    if mask.mean() < 0.5 or np.abs(warped.astype(float) - b)[mask].mean() > 15:
        return None
    return M.tolist()


def alignment(name: str, split: str, root: Path) -> np.ndarray | None:
    """Cached `_estimate_alignment` for one image; call `save_alignments` to keep new ones."""
    global _alignments
    if _alignments is None:
        path = root / ALIGNMENT_CACHE
        _alignments = orjson.loads(path.read_bytes()) if path.exists() else {}
    key = f"{split}/{name}"
    if key not in _alignments:
        _alignments[key] = _estimate_alignment(
            root / "xcorners" / split / "images" / f"{name}.jpg", root / "pieces" / split / "images" / f"{name}.jpg"
        )
    M = _alignments[key]
    return None if M is None else np.array(M)


def save_alignments(root: Path) -> None:
    if _alignments is not None:
        (root / ALIGNMENT_CACHE).write_bytes(orjson.dumps(_alignments))


AUTO_XCORNERS = "xcorners_auto.json"
XCORNER_REVIEW = "xcorners_review.json"
_auto_xcorners: dict[str, list] | None = None
_xcorner_review: dict[str, str] = {}


def auto_xcorners(name: str, split: str, root: Path) -> np.ndarray | None:
    """x-corners found by the xcorners model (auto_xcorners.py, since removed), already in the pieces
    image's pixels, for videos whose grid was checked and passed by review"""
    global _auto_xcorners, _xcorner_review
    if _auto_xcorners is None:
        path, review = root / AUTO_XCORNERS, root / XCORNER_REVIEW
        _auto_xcorners = orjson.loads(path.read_bytes()) if path.exists() else {}
        _xcorner_review = orjson.loads(review.read_bytes()) if review.exists() else {}
    if _xcorner_review.get(f"{split}/{source(name)}") != "ok":
        return None
    points = _auto_xcorners.get(f"{split}/{name}")
    return None if points is None else np.array(points)


def has_hand_xcorners(name: str, split: str, root: Path) -> bool:
    return (root / "xcorners" / split / "labels" / f"{name}.txt").exists()


def has_xcorners(name: str, split: str, root: Path) -> bool:
    return has_hand_xcorners(name, split, root) or auto_xcorners(name, split, root) is not None


def hand_xcorners(name: str, split: str, root: Path) -> np.ndarray | None:
    """The hand-labelled x-corners, mapped onto the pieces copy of the image"""
    M = alignment(name, split, root)
    if M is None:
        return None
    # The x-corners were labelled on the xcorners copy of the image, so map them onto the pieces copy
    with Image.open(root / "xcorners" / split / "images" / f"{name}.jpg") as im:
        x_width, x_height = im.size
    _, xboxes = read_yolo(root / "xcorners" / split / "labels" / f"{name}.txt", x_width, x_height)
    xcorners = np.stack([(xboxes[:, 0] + xboxes[:, 2]) / 2, (xboxes[:, 1] + xboxes[:, 3]) / 2], axis=1)
    return xcorners @ M[:, :2].T + M[:, 2]


def load_frame(name: str, split: str, root: Path, anchor_ratio: float, margin: float) -> Frame | None:
    image = root / "pieces" / split / "images" / f"{name}.jpg"
    with Image.open(image) as im:
        width, height = im.size
    # Hand labels win; frames without them (nyh) may have model-found corners
    if has_hand_xcorners(name, split, root):
        xcorners = hand_xcorners(name, split, root)
    else:
        xcorners = auto_xcorners(name, split, root)
    if xcorners is None or len(xcorners) != 49:
        return None
    H = fit_lattice(xcorners)
    if H is None:
        return None

    classes, boxes = read_yolo(root / "pieces" / split / "labels" / f"{name}.txt", width, height)
    # A piece stands on a point some fraction of its box width above the bottom of its box
    anchors = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3] - anchor_ratio * (boxes[:, 2] - boxes[:, 0])], axis=1
    )
    lattice = cv2.perspectiveTransform(anchors[None].astype(np.float32), H)[0] if len(anchors) else anchors
    squares = assign_squares(lattice, margin)
    on_board = squares[:, 0] >= 0

    prefix = source(name)
    group = name if prefix.startswith(PHOTO_PREFIXES) else f"{split}/{prefix}"
    return Frame(
        name=name,
        split=split,
        image=image,
        group=group,
        squares=squares[on_board],
        classes=classes[on_board],
        off_board=int((~on_board).sum()),
        H=H,
    )
