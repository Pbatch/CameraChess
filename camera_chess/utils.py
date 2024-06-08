import os
import shutil
from collections import namedtuple
from glob import glob

import chess.pgn
import cv2
import numpy as np
import yaml

from camera_chess.constants import DATA_DIR, BOARD_SIZE

video_config = namedtuple("VideoConfig", "start end url path keypoints fen moves roi")


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

    roi = d.get('roi', None)

    video_paths = list(glob(os.path.join(dataset_dir, 'video.*')))
    ext_priority = {'.mov': 1, '.mp4': 2, '.webm': 3}
    path = max(video_paths,
               key=lambda x: ext_priority[os.path.splitext(x)[1].lower()]
               )

    video_config_ = video_config(start=d['start'],
                                 end=d['end'],
                                 url=d['url'],
                                 path=path,
                                 keypoints=keypoints,
                                 fen=d['fen'],
                                 moves=load_moves_from_pgn(os.path.join(dataset_dir, 'gt.pgn')),
                                 roi=roi)

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


def update_state(state, update, decay=0.5):
    state *= decay
    state += (1 - decay) * update


def get_roi(keypoints, width, height, model_width, model_height, padding_ratio=12):
    x_min = np.min(keypoints[:, 0])
    x_max = np.max(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    y_max = np.max(keypoints[:, 1])

    roi_width = x_max - x_min
    roi_height = y_max - y_min
    padding_left = roi_width // padding_ratio
    padding_right = roi_width // padding_ratio
    padding_top = roi_height // padding_ratio
    padding_bottom = roi_height // padding_ratio

    padded_roi_width = roi_width + padding_left + padding_right
    padded_roi_height = roi_height + padding_top + padding_bottom
    ratio = padded_roi_height / padded_roi_width
    desired_ratio = model_height / model_width

    if ratio > desired_ratio:
        target_width = padded_roi_height / desired_ratio
        dx = target_width - padded_roi_width
        padding_left += dx // 2
        padding_right += dx - (dx // 2)
    else:
        target_height = padded_roi_width * desired_ratio
        padding_top += target_height - padded_roi_height

    roi = np.array([int(max(x_min - padding_left, 0)),
                    int(max(y_min - padding_top, 0)),
                    int(min(x_max + padding_right, width)),
                    int(min(y_max + padding_bottom, height))])
    return roi


def draw_points(d, xy, colour, radius=5):
    for x, y in xy:
        bbox = [x - radius, y - radius,
                x + radius, y + radius]
        d.ellipse(bbox, fill=colour)


def draw_lines(d, xy, colour, width=5):
    for i in range(len(xy)):
        d.line([*xy[i - 1], *xy[i]], fill=colour, width=width)


def draw_text(d, bbox, text, text_height=12):
    text_width = len(text) * 5
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
