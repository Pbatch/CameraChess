"""Local tool for labelling which corner is a1 on the independent photos (roboflow, google).

    uv run python -m camera_chess.labels.label_orientation   # then open http://127.0.0.1:8765

Labels are saved to orientation_labels.json as {name: rotation} (or "skip"), the rotation
meaning the same as in make_chessqueries_dataset.py, which reads the file.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import orjson
from loguru import logger
from tqdm import tqdm

from camera_chess.config import get_config
from camera_chess.labels.make_chessqueries_dataset import (
    CLASSES,
    PHOTO_PREFIXES,
    choose_rotation_by_model,
    is_excluded,
    load_frame,
    rotate,
    save_alignments,
)

ROOT = get_config().labels_root
LABELS = ROOT / "orientation_labels.json"


CORNER_SQUARES = {"a1": (0, 0), "h1": (7, 0), "h8": (7, 7), "a8": (0, 7)}


def board_corners(H: np.ndarray) -> list[dict[str, list[float]]]:
    """Pixel positions of the board's outer a1/h1/h8/a8 corners under each of the 4 rotations"""
    inv = np.linalg.inv(H)
    rotations = []
    for k in range(4):
        corners = {}
        for name, square in CORNER_SQUARES.items():
            f, r = rotate(np.array([square]), (4 - k) % 4)[0]
            p = inv @ np.array([-1 if f == 0 else 7, -1 if r == 0 else 7, 1.0])
            corners[name] = (p[:2] / p[2]).round(1).tolist()
        rotations.append(corners)
    return rotations


def build_items(predictions: dict) -> list[dict]:
    items = []
    for split in ("train", "val"):
        for path in tqdm(sorted((ROOT / "xcorners" / split / "labels").glob("*.txt")), desc=split):
            name = path.stem
            if is_excluded(name) or not name.startswith(PHOTO_PREFIXES):
                continue
            frame = load_frame(name, split, ROOT, 1 / 3, 0.5)
            # Only boards with both kings, one of each colour, are worth labelling
            if frame is None or any(
                np.sum(frame.classes == CLASSES.index(king)) != 1 for king in ("white-king", "black-king")
            ):
                continue
            k, margin = choose_rotation_by_model([frame], predictions)
            items.append(
                {
                    "name": name,
                    "image": f"img/{split}/{name}.jpg",
                    "corners": board_corners(frame.H),
                    "inv": np.linalg.inv(frame.H).tolist(),
                    "suggestion": k,
                    "margin": margin,
                }
            )
    save_alignments(ROOT)
    # The ones the model is least sure of first
    items.sort(key=lambda x: x["margin"])
    return items


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, items_json: bytes, **kwargs):
        self.items_json = items_json
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
            self._send((Path(__file__).parent / "label_orientation.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/items.json":
            self._send(self.items_json)
        elif path == "/labels.json":
            self._send(LABELS.read_bytes() if LABELS.exists() else b"{}")
        elif path.startswith("/img/"):
            split, name = path[len("/img/") :].split("/", 1)
            image = ROOT / "pieces" / split / "images" / Path(name).name
            if split in ("train", "val") and image.is_file():
                self._send(image.read_bytes(), "image/jpeg")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/labels":
            self.send_error(404)
            return
        labels = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tmp = LABELS.with_suffix(".tmp")
        tmp.write_text(json.dumps(labels, indent=1, sort_keys=True))
        tmp.replace(LABELS)
        self._send(b'{"ok": true}')

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    predictions = orjson.loads((ROOT / "chessq_predictions.json").read_bytes())
    items_json = orjson.dumps(build_items(predictions))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, items_json=items_json))
    logger.info(f"Labelling at http://127.0.0.1:{args.port} (labels in {LABELS})")
    server.serve_forever()


if __name__ == "__main__":
    main()
