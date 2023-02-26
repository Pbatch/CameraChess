import os
import shutil
from collections import namedtuple

import chess.pgn
import cv2
import numpy as np
import yaml

import constants

video_config = namedtuple("VideoConfig", "start end url path keypoints fen moves")


def get_square(idx):
    x = idx // 8
    y = idx % 8
    return f'{chr(x + 97)}{8 - y}'


def load_video_config(dataset):
    dataset_dir = os.path.join(constants.DATA_DIR, dataset)

    with open(os.path.join(constants.DATA_DIR, 'video_config.yaml')) as f:
        config = yaml.safe_load(f)
    d = config[dataset]
    if 'fen' not in d:
        # Starting position
        d['fen'] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    video_config_ = video_config(start=d['start'],
                                 end=d['end'],
                                 url=d['url'],
                                 path=os.path.join(dataset_dir, 'video.webm'),
                                 keypoints=np.array(d['keypoints'], dtype=np.float32),
                                 fen=d['fen'],
                                 moves=load_moves_from_pgn(os.path.join(dataset_dir, 'gt.pgn')))

    return video_config_


def load_moves_from_pgn(path):
    with open(path) as f:
        game = chess.pgn.read_game(f)
    moves = [str(move) for move in game.mainline_moves()]
    return moves


def clear_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)


def warp(src, keypoints):
    target = np.array([[constants.BOARD_SIZE, constants.BOARD_SIZE],
                       [0, constants.BOARD_SIZE],
                       [0, 0],
                       [constants.BOARD_SIZE, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(keypoints, target)
    inv_matrix = np.linalg.inv(matrix)
    warped_src = cv2.perspectiveTransform(np.expand_dims(src, axis=0), inv_matrix)[0]
    return warped_src
