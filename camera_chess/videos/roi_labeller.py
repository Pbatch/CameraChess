"""Serves a page for drawing each video's board boxes, and saves them to video_config.yaml.

`roi` is the 3D board and `overlay_roi` is the 2D board drawn over the video, both [x, y, width, height]
in video pixels. A video can instead be marked `unusable: true`, e.g. when the board can't be seen.

Usage: python -m camera_chess.videos.roi_labeller [--port 8000], then open http://localhost:8000
"""

import argparse
import json
import os
import re
from glob import glob
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import yaml
from loguru import logger

from camera_chess.constants import DATA_DIR
from camera_chess.videos.config import CONFIG_PATH, load_all

PAGE_PATH = os.path.join(os.path.dirname(__file__), "roi_labeller.html")
ROI_KEYS = ("roi", "overlay_roi")


def find_videos():
    """Datasets are the folders, two deep under data, that hold a video, e.g. yunker/Br8Y3kCJRCs"""
    config = load_all()
    videos = []
    for path in sorted(glob(os.path.join(DATA_DIR, "*", "*", "video.*"))):
        if path.endswith(".json"):
            continue
        dataset = os.path.relpath(os.path.dirname(path), DATA_DIR).replace(os.sep, "/")
        info_path = os.path.join(os.path.dirname(path), "video.info.json")
        title = dataset
        if os.path.isfile(info_path):
            with open(info_path) as f:
                title = json.load(f).get("title", dataset)
        d = config.get(dataset) or {}
        videos.append(
            {
                "dataset": dataset,
                "title": title,
                "path": path,
                "unusable": bool(d.get("unusable")),
                **{k: d.get(k) for k in ROI_KEYS},
            }
        )
    return videos


def save_labels(dataset, labels):
    config = load_all()
    d = config.setdefault(dataset, {})
    for k in ROI_KEYS:
        if labels.get(k) is not None:
            d[k] = [round(v) for v in labels[k]]
    if labels.get("unusable") is True:
        d["unusable"] = True
    elif labels.get("unusable") is False:
        d.pop("unusable", None)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info(f"Saved {dataset}: { ({k: d.get(k) for k in (*ROI_KEYS, 'unusable')}) }")


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self, path):
        # Browsers only seek in a video served with byte ranges
        size = os.path.getsize(path)
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:
                start = size - int(match.group(2))
        end = min(end, size - 1)
        self.send_response(206 if match else 200)
        self.send_header("Content-Type", "video/mp4" if path.endswith(".mp4") else "video/webm")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        if self.path == "/":
            with open(PAGE_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/videos":
            self._send_json([{k: v for k, v in video.items() if k != "path"} for video in find_videos()])
        elif self.path.startswith("/video/"):
            dataset = unquote(self.path[len("/video/") :])
            paths = {video["dataset"]: video["path"] for video in find_videos()}
            if dataset not in paths:
                self.send_error(404)
                return
            self._send_video(paths[dataset])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/labels":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["dataset"] not in {video["dataset"] for video in find_videos()}:
            self.send_error(404)
            return
        save_labels(body["dataset"], body)
        self._send_json({"ok": True})


def main(port):
    server = ThreadingHTTPServer(("localhost", port), Handler)
    logger.info(f"Labelling {len(find_videos())} videos at http://localhost:{port}, saving to {CONFIG_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8000)
    args = parser.parse_args()
    main(args.port)
