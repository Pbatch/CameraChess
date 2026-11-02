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
from pathlib import Path

import numpy as np
import orjson
from loguru import logger
from tqdm import tqdm

from camera_chess.config import get_config
from camera_chess.labels.frames import (
    CLASSES,
    Frame,
    agreement,
    apply_label_fixes,
    choose_rotation_by_model,
    has_hand_xcorners,
    has_xcorners,
    is_excluded,
    load_frame,
    rotate,
    save_alignments,
    to_fen,
)

WHITE_PAWN = CLASSES.index("white-pawn")
BLACK_PAWN = CLASSES.index("black-pawn")
# Hand verdicts on test labels from review_test_labels.py, oldest first
LABEL_REVIEWS = ("test_label_review.json", "test_label_review_dinov3.json")


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


def output_split(frame: Frame, test_fraction: float) -> str:
    """The frame's split in the dataset: CameraChess's val videos stay val, and a stable share of
    the other videos becomes test, whole videos at a time so no game is in two splits. Photos stay
    in train, as they aren't games."""
    if frame.split == "val" or "/" not in frame.group:
        return frame.split
    digest = hashlib.sha256(frame.group.encode()).digest()
    return "test" if int.from_bytes(digest[:8]) / 2**64 < test_fraction else frame.split


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=get_config().labels_root)
    parser.add_argument("--out", type=Path, default=get_config().DATA_ROOT / "chesscam")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.5,
        help="Drop groups whose best orientation beats the runner-up by less than this, per frame",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=get_config().labels_root / "chessq_predictions.json",
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
        default=get_config().labels_root / "orientation_labels.json",
        help="Hand labels from label_orientation.py, which override the chosen orientation",
    )
    parser.add_argument(
        "--second-opinion",
        type=Path,
        default=get_config().labels_root / "chessq_ft_predictions.json",
        help="Predictions from a model fine-tuned on this dataset, which can keep a group or frame "
        "that --predictions drops, outside the test split (which they would bias)",
    )
    parser.add_argument(
        "--label-reviews",
        type=Path,
        nargs="*",
        default=[get_config().labels_root / f for f in LABEL_REVIEWS],
        help='review_test_labels.py verdict files, oldest first; "model right" squares are relabelled',
    )
    args = parser.parse_args()
    label_fixes = load_label_fixes(args.label_reviews)
    manual = orjson.loads(args.orientation_labels.read_bytes()) if args.orientation_labels.exists() else {}
    predictions = orjson.loads(args.predictions.read_bytes()) if args.predictions.exists() else None
    logger.info(f"orienting with {'ChessQ Lite predictions' if predictions else 'chess priors'}")
    second = orjson.loads(args.second_opinion.read_bytes()) if predictions and args.second_opinion.exists() else {}
    if second:
        logger.info(f"second opinion on dropped train/val frames from {args.second_opinion.name}")

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
                    "corners": "hand" if has_hand_xcorners(frame.name, frame.split, args.root) else "auto",
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
    logger.info(f"wrote {len(annotations)} images {dict(by_split)} to {args.out}")
    logger.info(f"skipped {dict(skipped)}")
    save_alignments(args.root)


if __name__ == "__main__":
    main()
