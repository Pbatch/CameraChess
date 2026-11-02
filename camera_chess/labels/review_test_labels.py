"""Local tool for checking the ChessCam test labels where the fine-tuned model disagrees on one square.

    uv run python -m camera_chess.labels.review_test_labels   # then open http://127.0.0.1:8766

Reads the model's predictions from test_predictions_lite_ft_chesscam.json (written by
ChessQueries) and the dataset from data/chesscam. Verdicts are saved to
test_label_review.json as {name: {verdict, square, label, model}}, where the verdict is "label"
(the label is right), "model" (the model is right), "both" (both wrong) or "unsure".

With --chessq-preds (a preds_chesscam_test.json from scripts/eval_cross_dataset.py), every
square that model gets wrong is an item, verdicts are keyed "name:square" in --verdicts, and the
Lite verdicts above show as hints.

    uv run python -m camera_chess.labels.review_test_labels \\
        --chessq-preds outputs/chessqueries_dinov3-448-lr10x/preds_chesscam_test.json \\
        --chessq-probs outputs/chessqueries_dinov3-448-lr10x/probs_chesscam_test.json \\
        --verdicts test_label_review_dinov3.json

Add --min-p-model 0.9 to show only the squares that model is at least 90% sure of, and --by-video to
go through them one video at a time, in frame order, instead of most confident first.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import cv2
import numpy as np
import orjson
from loguru import logger
from PIL import Image
from tqdm import tqdm

from camera_chess.config import get_config
from camera_chess.labels.frames import (
    SYMBOLS,
    apply_label_fixes,
    assign_squares,
    load_frame,
    read_yolo,
    rotate,
    to_fen,
)

ROOT = get_config().labels_root
DATASET = get_config().DATA_ROOT / "chesscam"
PREDICTIONS = ROOT / "test_predictions_lite_ft_chesscam.json"
VERDICTS = ROOT / "test_label_review.json"
# The make_chessqueries_dataset.py defaults the dataset was built with
ANCHOR_RATIO, MARGIN = 1 / 3, 0.5


CHESSQ_SYMBOLS = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def squares_fen(squares: list[str]) -> str:
    """Inverse of fen_squares"""
    rows = []
    for r in range(8):
        row, empty = "", 0
        for s in squares[r * 8 : r * 8 + 8]:
            if s:
                row += (str(empty) if empty else "") + s
                empty = 0
            else:
                empty += 1
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)


def fen_squares(fen: str) -> list[str]:
    """64 symbols in FEN order (a8..h1), '' for empty"""
    return [c for row in fen.split("/") for ch in row for c in ([""] * int(ch) if ch.isdigit() else [ch])]


def square_polygon(inv: np.ndarray, index: int, k: int) -> list[list[float]]:
    """Pixel outline of a square, given by its FEN index in the dataset's orientation"""
    f, r = rotate(np.array([[index % 8, 7 - index // 8]]), (4 - k) % 4)[0]
    corners = np.array([[f - 1, r - 1], [f, r - 1], [f, r], [f - 1, r]], dtype=np.float32)
    return cv2.perspectiveTransform(corners[None], inv)[0].astype(float).round(1).tolist()


def piece_boxes(frame, k: int) -> list[dict]:
    """The labelled piece boxes, with the square (FEN index) each was put on, or None if off the board"""
    with Image.open(frame.image) as im:
        width, height = im.size
    classes, boxes = read_yolo(ROOT / "pieces" / frame.split / "labels" / f"{frame.name}.txt", width, height)
    if not len(boxes):
        return []
    anchors = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3] - ANCHOR_RATIO * (boxes[:, 2] - boxes[:, 0])], axis=1
    )
    lattice = cv2.perspectiveTransform(anchors[None].astype(np.float32), frame.H)[0]
    squares = assign_squares(lattice, MARGIN)
    result = []
    for c, box, square in zip(classes, boxes, squares):
        index = None
        if square[0] >= 0:
            f, r = rotate(square[None], k)[0]
            index = int((7 - r) * 8 + f)
        result.append({"box": box.round(1).tolist(), "symbol": SYMBOLS[c], "square": index})
    return result


def build_items(
    chessq_preds: Path | None = None,
    chessq_probs: Path | None = None,
    min_p_model: float | None = None,
    by_video: bool = False,
) -> list[dict]:
    predictions = orjson.loads(PREDICTIONS.read_bytes())
    annotations = {
        Path(a["image"]).stem: a
        for a in orjson.loads((DATASET / "annotations.json").read_bytes())
        if a["split"] == "test"
    }
    groups = orjson.loads((DATASET / "groups.json").read_bytes())
    chessq = orjson.loads(chessq_preds.read_bytes()) if chessq_preds else {}
    probs = orjson.loads(chessq_probs.read_bytes()) if chessq_probs else None
    prior = orjson.loads(VERDICTS.read_bytes()) if chessq and VERDICTS.exists() else {}
    items: list[dict[str, Any]] = []
    for name, p in tqdm(predictions.items()):
        label = fen_squares(annotations[name]["fen"])
        if chessq:
            model, second = [CHESSQ_SYMBOLS[c] for c in chessq[name]], fen_squares(p["finetuned"])
        else:
            model, second = fen_squares(p["finetuned"]), fen_squares(p["baseline"])
        diff = [i for i in range(64) if label[i] != model[i]]
        if not diff or (not chessq and len(diff) != 1):
            continue
        annotation = annotations[name]
        group = annotation["group"]
        report = groups[group]
        k = report.get("manual_rotation", report.get("model_rotation", report["prior_rotation"]))
        frame = load_frame(name, group.split("/")[0], ROOT, ANCHOR_RATIO, MARGIN)
        # The grid must reproduce the dataset's label, or the outlines would point at the wrong square
        rebuilt = (
            frame
            and apply_label_fixes(to_fen(rotate(frame.squares, k), frame.classes), annotation.get("fixed_squares", {}))[
                0
            ]
        )
        if frame is None or rebuilt != annotation["fen"]:
            raise ValueError(f"{name} no longer rebuilds to its dataset label")
        inv = np.linalg.inv(frame.H)
        a1, boxes = square_polygon(inv, 56, k), piece_boxes(frame, k)
        for index in diff:
            square = "abcdefgh"[index % 8] + str(8 - index // 8)
            # An earlier Lite verdict on this square and label, shown as a hint
            old = prior.get(name)
            hint = old if old and old["square"] == square and old["label"] == label[index] else None
            p = probs[name][index] if probs else None
            if min_p_model is not None and p is not None and p[chessq[name][index]] < min_p_model:
                continue
            items.append(
                {
                    "p_model": p[chessq[name][index]] if p else None,
                    "p_label": p[CHESSQ_SYMBOLS.index(label[index])] if p else None,
                    "key": f"{name}:{square}" if chessq else name,
                    "name": name,
                    "group": group,
                    "image": f"img/{annotation['image']}",
                    "index": index,
                    "square": square,
                    "diff": diff,
                    "hint": hint,
                    "label": label[index],
                    "model": model[index],
                    "baseline": second[index],
                    "label_fen": annotation["fen"],
                    "model_fen": squares_fen(model),
                    "outline": square_polygon(inv, index, k),
                    "a1": a1,
                    "boxes": boxes,
                }
            )
    # Frames of one game together and in order, as an error tends to persist across them
    items.sort(key=lambda x: (x["group"], int(x["name"].rsplit("_", 1)[1]), x["index"]))
    if probs and not by_video:
        # Most confident model first: the likeliest label errors
        items.sort(key=lambda x: -x["p_model"])
    return items


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, items_json: bytes, verdicts: Path, **kwargs):
        self.items_json = items_json
        self.verdicts = verdicts
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send(self, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?")[0])
        if path == "/":
            self._send((Path(__file__).parent / "review_test_labels.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/items.json":
            self._send(self.items_json)
        elif path == "/verdicts.json":
            self._send(self.verdicts.read_bytes() if self.verdicts.exists() else b"{}")
        elif path.startswith("/img/images/test/"):
            image = DATASET / "images" / "test" / Path(path).name
            if image.is_file():
                self._send(image.read_bytes(), "image/jpeg")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/verdicts":
            self.send_error(404)
            return
        verdicts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tmp = self.verdicts.with_suffix(".tmp")
        tmp.write_text(json.dumps(verdicts, indent=1, sort_keys=True))
        tmp.replace(self.verdicts)
        self._send(b'{"ok": true}')

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--chessq-preds",
        type=Path,
        default=None,
        help="Review every wrong square of these chessqueries predictions instead",
    )
    parser.add_argument(
        "--chessq-probs",
        type=Path,
        default=None,
        help="Per-square softmax for --chessq-preds; sorts by the model's confidence",
    )
    parser.add_argument(
        "--min-p-model",
        type=float,
        default=None,
        help="Only squares the --chessq-probs model is at least this confident in",
    )
    parser.add_argument(
        "--by-video",
        action="store_true",
        help="Keep frames of one video together and in order, rather than sorting by confidence",
    )
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    args = parser.parse_args()
    if args.chessq_preds and args.verdicts.resolve() == VERDICTS.resolve():
        parser.error("--chessq-preds needs its own --verdicts file")
    if args.chessq_probs and not args.chessq_preds:
        parser.error("--chessq-probs needs --chessq-preds")
    if args.min_p_model is not None and not args.chessq_probs:
        parser.error("--min-p-model needs --chessq-probs")
    items = build_items(args.chessq_preds, args.chessq_probs, args.min_p_model, args.by_video)
    handler = partial(Handler, items_json=orjson.dumps(items), verdicts=args.verdicts)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    logger.info(f"Reviewing {len(items)} squares at http://127.0.0.1:{args.port} (verdicts in {args.verdicts})")
    server.serve_forever()


if __name__ == "__main__":
    main()
