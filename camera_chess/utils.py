import os
import shutil
from collections import namedtuple

import chess.pgn
import numpy as np
import yaml

from camera_chess.constants import DATA_DIR

video_config = namedtuple("VideoConfig", "start end url path keypoints fen moves")


def get_square(idx):
    x = idx // 8
    y = idx % 8
    return f'{chr(x + 97)}{8 - y}'


def load_video_config(dataset):
    dataset_dir = os.path.join(DATA_DIR, dataset)

    with open(os.path.join(DATA_DIR, 'video_config.yaml')) as f:
        config = yaml.safe_load(f)
    dataset_config = config[dataset]

    dataset_config['keypoints'] = np.array(dataset_config['keypoints'], dtype=np.float32)
    dataset_config['path'] = os.path.join(dataset_dir, dataset_config['path'])
    pgn_path = os.path.join(dataset_dir, 'gt.pgn')
    dataset_config['moves'] = load_moves_from_pgn(pgn_path)
    dataset_config = video_config(*dataset_config.values())

    return dataset_config


def load_moves_from_pgn(path):
    with open(path) as f:
        game = chess.pgn.read_game(f)
    moves = [str(move) for move in game.mainline_moves()]
    return moves


def clear_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)

