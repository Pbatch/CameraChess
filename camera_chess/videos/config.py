"""Each video's settings in data/video_config.yaml, keyed by dataset (e.g. yunker/Br8Y3kCJRCs), and its video file."""

import os

import yaml

from camera_chess.constants import DATA_DIR

CONFIG_PATH = os.path.join(DATA_DIR, "video_config.yaml")


def load_all():
    if not os.path.isfile(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def load_config(dataset):
    return load_all()[dataset]


def video_path(dataset):
    d = os.path.join(DATA_DIR, dataset)
    return os.path.join(d, next(p for p in sorted(os.listdir(d)) if p.startswith("video.") and not p.endswith(".json")))
