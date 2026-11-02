"""Local tool for checking the model-found x-corners (xcorners_auto.json), one video at a time.

    uv run python -m camera_chess.labels.review_xcorners   # then open http://127.0.0.1:8766

Each video shows its first and last frame with the grid drawn on, so a grid that is wrong, or
drifts off the board, can be rejected. Verdicts are saved to xcorners_review.json as
{video: "ok" | "bad"}, and make_chessqueries_dataset.py only uses auto corners from "ok" videos.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import orjson

from camera_chess.constants import LABELS_DIR
from camera_chess.labels.make_chessqueries_dataset import AUTO_XCORNERS, fit_lattice, source

ROOT = Path(LABELS_DIR)
REVIEW = ROOT / "xcorners_review.json"


def build_items() -> list[dict]:
    auto = orjson.loads((ROOT / AUTO_XCORNERS).read_bytes())
    videos = defaultdict(list)
    for key in auto:
        split, name = key.split("/")
        videos[f"{split}/{source(name)}"].append((int(name.rsplit("_", 1)[1]), split, name))
    items = []
    for video, frames in sorted(videos.items()):
        frames.sort()
        shown = []
        for _, split, name in (frames[0], frames[-1]) if len(frames) > 1 else frames[:1]:
            H = fit_lattice(np.array(auto[f"{split}/{name}"]))
            if H is not None:
                shown.append({"image": f"img/{split}/{name}.jpg", "inv": np.linalg.inv(H).tolist()})
        if shown:
            items.append({"video": video, "frames": len(frames), "shown": shown})
    return items


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, items_json: bytes, **kwargs):
        self.items_json = items_json
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send(self, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?")[0])
        if path == "/":
            self._send((Path(__file__).parent / "review_xcorners.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/items.json":
            self._send(self.items_json)
        elif path == "/review.json":
            self._send(REVIEW.read_bytes() if REVIEW.exists() else b"{}")
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
        if self.path != "/review":
            self.send_error(404)
            return
        review = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tmp = REVIEW.with_suffix(".tmp")
        tmp.write_text(json.dumps(review, indent=1, sort_keys=True))
        os.replace(tmp, REVIEW)
        self._send(b'{"ok": true}')

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    items_json = orjson.dumps(build_items())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, items_json=items_json))
    print(f"Reviewing at http://127.0.0.1:{args.port} (verdicts in {REVIEW})")
    server.serve_forever()


if __name__ == "__main__":
    main()
