"""Each video's settings in data/video_config.yaml, keyed by dataset (e.g. yunker/Br8Y3kCJRCs), and its video file."""

import yaml

from camera_chess.config import get_config

CONFIG_PATH = get_config().DATA_ROOT / "video_config.yaml"


def load_all():
    if not CONFIG_PATH.is_file():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def load_config(dataset):
    return load_all()[dataset]


def video_path(dataset):
    d = get_config().DATA_ROOT / dataset
    return next(p for p in sorted(d.glob("video.*")) if p.suffix != ".json")
