"""Each video's settings in data/video_config.yaml, keyed by dataset (e.g. yunker/Br8Y3kCJRCs), and its video file."""

import yaml

from camera_chess.config import get_config


def config_path():
    return get_config().DATA_ROOT / "video_config.yaml"


def load_all():
    path = config_path()
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def save_all(config):
    config_path().write_text(yaml.safe_dump(config, sort_keys=False))


def load_config(dataset):
    return load_all()[dataset]


def video_path(dataset):
    d = get_config().DATA_ROOT / dataset
    return next(p for p in sorted(d.glob("video.*")) if p.suffix != ".json")
