"""Convert the YOLO pieces + xcorners labels into a ChessQueries dataset (one FEN per image).

Only images with both labels can be converted, since the 49 x-corners are what place the
8x8 grid. The grid's orientation (which corner is a1) is not labelled, so it is chosen per
video: by agreement with ChessQ Lite's predictions when `predict_chessq.py` has been run,
otherwise with chess priors (pawns never stand on a back rank, and White's pieces sit lower
on the board than Black's). Only the orientation comes from the model; the pieces are the
human labels.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import orjson
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from camera_chess.constants import DATA_DIR, LABELS_DIR

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
WHITE_PAWN = CLASSES.index("white-pawn")
BLACK_PAWN = CLASSES.index("black-pawn")
# Duplicates ChessReD, so would leak its test set into training
EXCLUDED_PREFIXES = ("chessred2k",)
# Independent photos rather than frames of one video, so each gets its own orientation
PHOTO_PREFIXES = ("roboflow", "google")
# Photo sources whose every photo was skipped when labelling orientation by hand
EXCLUDED_SOURCES = {"roboflow_11", "roboflow_3", "roboflow_4", "roboflow_6", "roboflow_final_chess_piece"}
# Hand verdicts on test labels from review_test_labels.py, oldest first
LABEL_REVIEWS = ("test_label_review.json", "test_label_review_dinov3.json")


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
    rotation is arbitrary until `choose_rotation` fixes it."""
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


def orientation_penalty(frames: list[Frame], k: int) -> float:
    penalty = 0.0
    for frame in frames:
        if len(frame.classes) == 0:
            continue
        ranks = rotate(frame.squares, k)[:, 1]
        white = frame.classes >= 6
        # Pawns can't be on either back rank (they start on the 2nd/7th and promote on the 8th/1st)
        pawns = (frame.classes == WHITE_PAWN) | (frame.classes == BLACK_PAWN)
        penalty += 10 * np.sum(pawns & ((ranks == 0) | (ranks == 7)))
        if white.any() and (~white).any():
            penalty += ranks[white].mean() - ranks[~white].mean()
    return penalty


def choose_rotation(frames: list[Frame]) -> tuple[int, float]:
    """The rotation putting a1 bottom-left, and how far ahead of the runner-up it scores."""
    penalties = sorted((orientation_penalty(frames, k), k) for k in range(4))
    return penalties[0][1], (penalties[1][0] - penalties[0][0]) / max(1, len(frames))


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


def output_split(frame: Frame, test_fraction: float) -> str:
    """The frame's split in the dataset: CameraChess's val videos stay val, and a stable share of
    the other videos becomes test, whole videos at a time so no game is in two splits. Photos stay
    in train, as they aren't games."""
    if frame.split == "val" or "/" not in frame.group:
        return frame.split
    digest = hashlib.sha256(frame.group.encode()).digest()
    return "test" if int.from_bytes(digest[:8]) / 2**64 < test_fraction else frame.split


def to_fen(squares: np.ndarray, classes: np.ndarray) -> str:
    grid = [["" for _ in range(8)] for _ in range(8)]
    for (f, r), c in zip(squares, classes):
        grid[7 - r][f] = SYMBOLS[c]
    return grid_fen(grid)


def grid_fen(grid: list[list[str]]) -> str:
    """FEN placement of 8 rows (rank 8 first) of symbols, '' for empty"""
    rows = []
    for row in grid:
        text, empty = "", 0
        for piece in row:
            if piece:
                text += (str(empty) if empty else "") + piece
                empty = 0
            else:
                empty += 1
        rows.append(text + (str(empty) if empty else ""))
    return "/".join(rows)


def load_label_fixes(paths: list[Path]) -> dict[str, dict[str, list[str]]]:
    """name -> {square: [label, correction]} from review_test_labels.py verdicts that the model was right.

    Later files win where they judge the same square: "label" or "both" there undoes an earlier fix."""
    fixes: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for path in paths:
        if not path.exists():
            continue
        for key, v in orjson.loads(path.read_bytes()).items():
            name = key.split(":")[0]
            if v["verdict"] == "model":
                fixes[name][v["square"]] = [v["label"], v["model"]]
            elif v["verdict"] in ("label", "both"):
                fixes[name].pop(v["square"], None)
    return fixes


def apply_label_fixes(fen: str, fixes: dict[str, list[str]]) -> tuple[str, dict[str, list[str]]]:
    """The FEN with each fix applied, and the fixes used. A fix only applies while its square still holds
    the label that was reviewed, so a relabelled square isn't overwritten."""
    grid = [[c for ch in row for c in ([""] * int(ch) if ch.isdigit() else [ch])] for row in fen.split("/")]
    used = {}
    for square, (label, correction) in fixes.items():
        r, f = 8 - int(square[1]), "abcdefgh".index(square[0])
        if grid[r][f] == label:
            grid[r][f] = correction
            used[square] = [label, correction]
    return grid_fen(grid), used


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
    """x-corners found by auto_xcorners.py, already in the pieces image's pixels, for videos whose
    grid was checked and passed in review_xcorners.py"""
    global _auto_xcorners, _xcorner_review
    if _auto_xcorners is None:
        path, review = root / AUTO_XCORNERS, root / XCORNER_REVIEW
        _auto_xcorners = orjson.loads(path.read_bytes()) if path.exists() else {}
        _xcorner_review = orjson.loads(review.read_bytes()) if review.exists() else {}
    if _xcorner_review.get(f"{split}/{source(name)}") != "ok":
        return None
    points = _auto_xcorners.get(f"{split}/{name}")
    return None if points is None else np.array(points)


def has_xcorners(name: str, split: str, root: Path) -> bool:
    return (root / "xcorners" / split / "labels" / f"{name}.txt").exists() or auto_xcorners(
        name, split, root
    ) is not None


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
    if (root / "xcorners" / split / "labels" / f"{name}.txt").exists():
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(LABELS_DIR))
    parser.add_argument("--out", type=Path, default=Path(DATA_DIR) / "chesscam")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.5,
        help="Drop groups whose best orientation beats the runner-up by less than this, per frame",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(LABELS_DIR) / "chessq_predictions.json",
        help="ChessQ Lite predictions from predict_chessq.py, used to orient each group",
    )
    parser.add_argument(
        "--min-model-margin",
        type=float,
        default=4,
        help="With predictions, drop groups whose best orientation leads by fewer squares per frame",
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=48,
        help="With predictions, drop frames agreeing with the model on fewer squares, "
        "as the grid or labels are likely wrong",
    )
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=1 / 3,
        help="How far above the bottom of its box a piece stands, as a fraction of box width",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="How far outside the board, in squares, an anchor can fall and still count as on it",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.12, help="Share of videos (not frames) held out as the test split"
    )
    parser.add_argument(
        "--orientation-labels",
        type=Path,
        default=Path(LABELS_DIR) / "orientation_labels.json",
        help="Hand labels from label_orientation.py, which override the chosen orientation",
    )
    parser.add_argument(
        "--second-opinion",
        type=Path,
        default=Path(LABELS_DIR) / "chessq_ft_predictions.json",
        help="Predictions from a model fine-tuned on this dataset, which can keep a group or frame "
        "that --predictions drops, outside the test split (which they would bias)",
    )
    parser.add_argument(
        "--label-reviews",
        type=Path,
        nargs="*",
        default=[Path(LABELS_DIR) / f for f in LABEL_REVIEWS],
        help='review_test_labels.py verdict files, oldest first; "model right" squares are relabelled',
    )
    args = parser.parse_args()
    label_fixes = load_label_fixes(args.label_reviews)
    manual = orjson.loads(args.orientation_labels.read_bytes()) if args.orientation_labels.exists() else {}
    predictions = orjson.loads(args.predictions.read_bytes()) if args.predictions.exists() else None
    print("orienting with", "ChessQ Lite predictions" if predictions else "chess priors")
    second = orjson.loads(args.second_opinion.read_bytes()) if predictions and args.second_opinion.exists() else {}
    if second:
        print(f"second opinion on dropped train/val frames from {args.second_opinion.name}")

    frames: list[Frame] = []
    skipped = defaultdict(int)
    for split in ("train", "val"):
        names = sorted(p.stem for p in (args.root / "pieces" / split / "labels").glob("*.txt"))
        for name in tqdm(names, desc=split):
            if is_excluded(name):
                skipped["excluded"] += 1
            elif not has_xcorners(name, split, args.root):
                skipped["no xcorners"] += 1
            elif (frame := load_frame(name, split, args.root, args.anchor_ratio, args.margin)) is None:
                skipped["alignment or grid fit failed"] += 1
            else:
                frames.append(frame)

    groups: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        groups[frame.group].append(frame)

    if args.out.exists():
        shutil.rmtree(args.out)
    annotations = []
    group_report = {}
    for group, group_frames in sorted(groups.items()):
        prior_k, prior_margin = choose_rotation(group_frames)
        report = {"frames": len(group_frames), "prior_rotation": prior_k, "prior_margin": round(float(prior_margin), 3)}
        if predictions is None:
            k, clear = prior_k, prior_margin >= args.min_margin
        else:
            k, model_margin = choose_rotation_by_model(group_frames, predictions)
            report.update(model_rotation=k, model_margin=round(model_margin, 2))
            clear = model_margin >= args.min_model_margin
        # Photos are their own group, so a hand label for the photo orients the whole group
        if group in manual:
            if manual[group] == "skip":
                skipped["skipped by labeller"] += len(group_frames)
                continue
            k, clear = manual[group], True
            report["manual_rotation"] = k
        # A group's frames share a split, as whole videos go to test
        test = output_split(group_frames[0], args.test_fraction) == "test"
        if not clear and not test and second and all(f.name in second for f in group_frames):
            k, second_margin = choose_rotation_by_model(group_frames, second)
            report.update(second_opinion_rotation=k, second_opinion_margin=round(second_margin, 2))
            clear = second_margin >= args.min_model_margin
            if clear:
                skipped["(orientation kept by second opinion)"] += len(group_frames)
        group_report[group] = report
        if not clear:
            skipped["orientation unclear"] += len(group_frames)
            continue
        for frame in group_frames:
            second_agreement = None
            if predictions is not None:
                # A hand label can overrule the model's orientation, so check the grid and pieces against the
                # model at whichever rotation it reads the board in, rather than at the hand-labelled one
                rotations = range(4) if group in manual else [k]
                frame_agreement = max(agreement(frame, r, predictions[frame.name]) for r in rotations)
                if frame_agreement < args.min_agreement and not test and frame.name in second:
                    second_agreement = max(agreement(frame, r, second[frame.name]) for r in rotations)
                if frame_agreement < args.min_agreement and (second_agreement or 0) < args.min_agreement:
                    skipped["disagrees with model"] += 1
                    continue
                if second_agreement is not None:
                    skipped["(frame kept by second opinion)"] += 1
            split = output_split(frame, args.test_fraction)
            dest = args.out / "images" / split / frame.image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frame.image, dest)
            fen, fixed = apply_label_fixes(
                to_fen(rotate(frame.squares, k), frame.classes), label_fixes.get(frame.name, {})
            )
            skipped["(squares relabelled by review)"] += len(fixed)
            annotations.append(
                {
                    "image": str(dest.relative_to(args.out)),
                    "fen": fen,
                    **({"fixed_squares": fixed} if fixed else {}),
                    "split": split,
                    "group": group,
                    "off_board": frame.off_board,
                    "corners": "hand"
                    if (args.root / "xcorners" / frame.split / "labels" / f"{frame.name}.txt").exists()
                    else "auto",
                    "model_agreement": frame_agreement if predictions is not None else None,
                    **({"second_opinion_agreement": second_agreement} if second_agreement is not None else {}),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "annotations.json").write_bytes(orjson.dumps(annotations, option=orjson.OPT_INDENT_2))
    (args.out / "groups.json").write_bytes(orjson.dumps(group_report, option=orjson.OPT_INDENT_2))
    by_split = defaultdict(int)
    for a in annotations:
        by_split[a["split"]] += 1
    print(f"wrote {len(annotations)} images {dict(by_split)} to {args.out}")
    print("skipped", dict(skipped))
    save_alignments(args.root)


if __name__ == "__main__":
    main()
