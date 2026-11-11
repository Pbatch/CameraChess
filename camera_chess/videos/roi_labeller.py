"""Serves a page for drawing each video's board boxes, and saves them to video_config.yaml.

`roi` is the 3D board as [x, y, width, height] in video pixels. The 2D board (`overlay_roi`) is found by a
script, so this tool only shows it and never saves it. A video can instead be marked `unusable: true`, e.g. when the board can't be seen.

Usage: python -m camera_chess.videos.roi_labeller [--port 8000], then open http://localhost:8000
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from camera_chess.config import get_config
from camera_chess.local_server import LocalHandler, serve
from camera_chess.videos.config import config_path, load_all, save_all

PAGE_PATH = Path(__file__).with_name("roi_labeller.html")
ROI_KEYS = ("roi",)


def find_videos():
    """Datasets are the folders, two deep under data, that hold a video, e.g. yunker/Br8Y3kCJRCs"""
    config = load_all()
    videos = []
    data_root = get_config().DATA_ROOT
    for path in sorted(data_root.glob("*/*/video.*")):
        if path.suffix == ".json":
            continue
        dataset = path.parent.relative_to(data_root).as_posix()
        info_path = path.with_name("video.info.json")
        title = dataset
        if info_path.is_file():
            title = json.loads(info_path.read_text()).get("title", dataset)
        d = config.get(dataset) or {}
        videos.append(
            {
                "dataset": dataset,
                "title": title,
                "path": path,
                "unusable": bool(d.get("unusable")),
                **{k: d.get(k) for k in (*ROI_KEYS, "overlay_roi")},
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
    save_all(config)
    logger.info(f"Saved {dataset}: { ({k: d.get(k) for k in (*ROI_KEYS, 'unusable')}) }")


class Handler(LocalHandler):
    def do_GET(self):
        path = self.route
        if path == "/":
            self.send_page(PAGE_PATH)
        elif path == "/api/videos":
            self.send_json([{k: v for k, v in video.items() if k != "path"} for video in find_videos()])
        elif path.startswith("/video/"):
            dataset = path[len("/video/") :]
            paths = {video["dataset"]: video["path"] for video in find_videos()}
            if dataset not in paths:
                self.send_error(404)
                return
            self.send_video(paths[dataset])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.route != "/api/labels":
            self.send_error(404)
            return
        body = self.read_json()
        if body["dataset"] not in {video["dataset"] for video in find_videos()}:
            self.send_error(404)
            return
        save_labels(body["dataset"], body)
        self.send_json({"ok": True})


def main(port):
    serve(Handler, port, f"Labelling {len(find_videos())} videos, saving to {config_path()}", host="localhost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8000)
    args = parser.parse_args()
    main(args.port)
