"""Each video's settings in data/video_config.yaml, keyed by dataset (e.g. yunker/Br8Y3kCJRCs), and its video file."""

import yaml
from loguru import logger

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


def is_unusable(dataset, config=None):
    """Whether the dataset, or for `<id>_game_N` its whole video, is marked `unusable: true` (e.g. by roi_labeller)"""
    config = load_all() if config is None else config
    return any((config.get(d) or {}).get("unusable") for d in {dataset, dataset.split("_game_")[0]})


def usable(datasets):
    """The datasets not marked unusable, in order, logging any left out"""
    config = load_all()
    kept = [d for d in datasets if not is_unusable(d, config)]
    if skipped := [d for d in datasets if d not in kept]:
        logger.warning(f"Skipping {skipped}: marked unusable in video_config.yaml")
    return kept


def load_config(dataset):
    return load_all()[dataset]


def video_path(dataset):
    d = get_config().DATA_ROOT / dataset
    return next(p for p in sorted(d.glob("video.*")) if p.suffix != ".json")
