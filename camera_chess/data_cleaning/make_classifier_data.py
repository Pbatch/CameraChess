import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR, CLASSIFIER_DIR, BOARD_SIZE, SQUARE_SIZE, MARGIN
from camera_chess.utils import clear_dir, warp
from tracker import _xy_to_square


def warp_chessboard_image(img, src):
    dst_points = np.array([[MARGIN, MARGIN],
                           [BOARD_SIZE + MARGIN, MARGIN],
                           [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
                           [MARGIN, BOARD_SIZE + MARGIN]],
                          dtype=np.float32)
    transformation_matrix, mask = cv2.findHomography(src, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (BOARD_SIZE + 2 * MARGIN, BOARD_SIZE + 2 * MARGIN))


DATASETS = {
    # 'synthetic': ['chesscog',
    #               *[os.path.join('roboflow', s) for s in ['7', '10']]
    #               ],
    'val': ['google',
            os.path.join('peter', 'scholars_mate'),
            os.path.join('youtube', 'carlsen_vidit')],
    'train': ['google_empty',
              *[os.path.join('youtube', s) for s in ['hikaru_sarin', 'dubov_nepo', 'carlsen_toma', 'shimanov_vidit',
                                                     'harika_nana', 'anand_carlsen', 'gukesh_shakh',
                                                     'karayaman', 'hikaru_vasif', 'magnus_madaminov', 'hari_tuan',
                                                     'hans_rinat', 'retired_lawyer', 'ramirez_yoo']],
              *[os.path.join('peter', s) for s in ['smothered_mate', 'kasparov_immortal', 'peter_emma',
                                                   'wells_shirov', 'gerasimov_smyslov', 'bronstein_teschner',
                                                   'melgosa_zuluaga', 'campora_morozevich', 'eingorn_vaganian',
                                                   'tal_sviridov', 'larsen_spassky', 'furman_spassky',
                                                   'wells_speelman', 'glass_1', 'glass_2', 'glass_3', 'pub']],
              *[os.path.join('single_piece', s) for s in CLASSES],
              *[os.path.join('roboflow', s) for s in ['1', '2', '3', '4', '5', '6', '8', '9', '11',
                                                      'public', 'ppp']],
              *[os.path.join('peter_wooden', s) for s in ['scholars_mate', 'smothered_mate', 'gerasimov_smyslov',
                                                          'wells_shirov']]
              ]
}


def get_nearest_square(square_centers, center):
    idx = np.argmin(np.sum((center - square_centers) ** 2, axis=1))
    x = idx // 8
    y = idx % 8
    return f'{chr(x + 97)}{8 - y}'


def square_to_xy(square):
    x = ord(square[0]) - 97
    y = 8 - int(square[1])
    return x, y


def process_dataset(split, dataset, image_size):
    label_paths = list(glob(os.path.join(DATA_DIR, dataset, 'labels', '*')))
    for label_path in tqdm(label_paths, desc=dataset):
        with open(label_path, 'r') as f:
            label = json.load(f)
        if 'keypoints' not in label:
            continue

        image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
        id_ = f'{dataset.replace(os.path.sep, "_")}_{os.path.splitext(os.path.basename(image_path))[0]}'
        image = Image.open(image_path)

        keypoints = np.array(list(label['keypoints'].values()))
        keypoints[:, 0] *= image.width
        keypoints[:, 1] *= image.height

        pieces = [bbox[0] for bbox in label['bboxes']]
        bboxes = np.array([bbox[1:] for bbox in label['bboxes']])
        if len(bboxes):
            bboxes[..., [0, 2]] *= image.width
            bboxes[..., [1, 3]] *= image.height

        grid = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        square_centers = warp(grid, keypoints.astype(np.float32))
        square_to_piece = {_xy_to_square(i // 8, i % 8): 'empty' for i in range(64)}
        for piece, bbox in zip(pieces, bboxes):
            center = np.array([bbox[0] + bbox[2] / 2,
                               (bbox[1] + bbox[3] - bbox[2] / 4)])
            square = get_nearest_square(square_centers, center)
            square_to_piece[square] = piece

        pair_sums = [keypoints[-i][1] + keypoints[-i + 1][1] for i in range(4)]
        shift = np.argmin(pair_sums)
        keypoints = np.roll(keypoints, shift, axis=0)
        warped_image = warp_chessboard_image(np.array(image), keypoints)

        for square, piece in square_to_piece.items():
            x, y = square_to_xy(square)
            # Move h1 to top-left
            x, y = 7 - x, 7 - y
            # Clockwise rotation of 90 degrees
            for _ in range(shift):
                x, y = 7 - y, x

            x1 = int(MARGIN + SQUARE_SIZE * (x - 1))
            x2 = int(MARGIN + SQUARE_SIZE * (x + 2))
            y1 = int(MARGIN + SQUARE_SIZE * (y - 2))
            y2 = int(MARGIN + SQUARE_SIZE * (y + 1))
            crop = Image.fromarray(warped_image[y1:y2, x1:x2]).resize((image_size, image_size))

            new_image_path = os.path.join(CLASSIFIER_DIR, split, f'{id_}_{square}_{piece}.jpg')
            crop.save(new_image_path)


def main(image_size):
    for split, datasets in DATASETS.items():
        clear_dir(os.path.join(CLASSIFIER_DIR, split))
        for dataset in datasets:
            process_dataset(split, dataset, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--image_size', type=int, default=256)
    args = parser.parse_args()
    main(args.image_size)
