import os
import pickle
import shutil
from collections import namedtuple
from glob import glob

import chess.pgn
import cv2
import numpy as np
import yaml
from PIL import ImageFont

from camera_chess.constants import DATA_DIR, BOARD_SIZE

video_config = namedtuple("VideoConfig", "start end url path keypoints fen moves")


def get_square(idx):
    x = idx // 8
    y = idx % 8
    return f'{chr(x + 97)}{8 - y}'


def load_video_config(dataset):
    dataset_dir = os.path.join(DATA_DIR, dataset)

    with open(os.path.join(DATA_DIR, 'video_config.yaml')) as f:
        config = yaml.safe_load(f)
    d = config[dataset.replace('\\', '/')]
    if 'fen' not in d:
        # Starting position
        d['fen'] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    if 'url' not in d:
        d['url'] = 'n/a'

    if 'keypoints' in d:
        keypoints = np.array(d['keypoints'], dtype=np.float32)
    else:
        keypoints = None

    video_config_ = video_config(start=d['start'],
                                 end=d['end'],
                                 url=d['url'],
                                 path=glob(os.path.join(dataset_dir, 'video.*'))[0],
                                 keypoints=keypoints,
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
    target = np.array([[BOARD_SIZE, BOARD_SIZE],
                       [0, BOARD_SIZE],
                       [0, 0],
                       [BOARD_SIZE, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(keypoints, target)
    inv_matrix = np.linalg.inv(matrix)
    warped_src = cv2.perspectiveTransform(np.expand_dims(src, axis=0), inv_matrix)[0]
    return warped_src


def serialize(obj):
    return pickle.dumps(obj).decode("ISO-8859-1")


def deserialize(s):
    return pickle.loads(s.encode("ISO-8859-1"))


def update_state(state, update, decay=0.5):
    state *= decay
    state += (1 - decay) * update


def draw_text(d, bbox, text):
    font = ImageFont.load_default()
    text_width, text_height = font.getsize(text)
    y_offset = -10
    x = (bbox[0] + bbox[2] - text_width) / 2
    y = bbox[1] - y_offset

    mid_x = (bbox[0] + bbox[2]) / 2
    text_bbox = (mid_x - text_width / 2 - 5,
                 bbox[1] - y_offset - text_height,
                 mid_x + text_width / 2 + 5,
                 bbox[1] - y_offset)
    d.rectangle(text_bbox,
                fill='black')
    d.rectangle(tuple(bbox))
    d.text((x, y - text_height), text=text)


def draw_points(d, xy, colour, radius=5):
    for x, y in xy:
        bbox = [x - radius, y - radius,
                x + radius, y + radius]
        d.ellipse(bbox, fill=colour)


def draw_lines(d, xy, colour, width=5):
    for i in range(len(xy)):
        d.line([*xy[i-1], *xy[i]], fill=colour, width=width)
