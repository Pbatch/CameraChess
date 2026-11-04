"""Local tool for labelling which corner is a1 on the independent photos (roboflow, google).

    uv run python -m camera_chess.labels.label_orientation   # then open http://127.0.0.1:8765

Labels are saved to orientation_labels.json as {name: rotation} (or "skip"), the rotation
meaning the same as in make_chessqueries_dataset.py, which reads the file.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

from camera_chess.config import get_config
from camera_chess.labels.frames import (
    CLASSES,
    PHOTO_PREFIXES,
    choose_rotation_by_model,
    is_excluded,
    load_frame,
    rotate,
    save_alignments,
)
from camera_chess.local_server import LocalHandler, save_json, serve

# In the labels folder
LABELS = "orientation_labels.json"


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


def build_items(predictions: dict, root: Path) -> list[dict]:
    items = []
    for split in ("train", "val"):
        for path in tqdm(sorted((root / "xcorners" / split / "labels").glob("*.txt")), desc=split):
            name = path.stem
            if is_excluded(name) or not name.startswith(PHOTO_PREFIXES):
                continue
            frame = load_frame(name, split, root, 1 / 3, 0.5)
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
    save_alignments(root)
    # The ones the model is least sure of first
    items.sort(key=lambda x: x["margin"])
    return items


class Handler(LocalHandler):
    def __init__(self, *args, items_json: bytes, root: Path, **kwargs):
        self.items_json = items_json
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = self.route
        labels = self.root / LABELS
        if path == "/":
            self.send_page(Path(__file__).with_name("label_orientation.html"))
        elif path == "/items.json":
            self.send(self.items_json)
        elif path == "/labels.json":
            self.send(labels.read_bytes() if labels.exists() else b"{}")
        elif path.startswith(("/img/train/", "/img/val/")):
            split, name = path[len("/img/") :].split("/", 1)
            self.send_file(self.root / "pieces" / split / "images" / Path(name).name, "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.route != "/labels":
            self.send_error(404)
            return
        save_json(self.root / LABELS, self.read_json())
        self.send_json({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=get_config().labels_root)
    args = parser.parse_args()
    predictions = orjson.loads((args.root / "chessq_predictions.json").read_bytes())
    items_json = orjson.dumps(build_items(predictions, args.root))
    serve(
        partial(Handler, items_json=items_json, root=args.root),
        args.port,
        f"Labelling (labels in {args.root / LABELS})",
    )


if __name__ == "__main__":
    main()
