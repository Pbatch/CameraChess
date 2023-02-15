import json
import os

import chess
import cv2
import numpy as np
import scipy.spatial
from PIL import Image

from chess import Board
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from camera_chess.constants import LABEL_STUDIO_DIR, UPLOAD_DIR, ROOT_DIR, CORNERS, CLASS_TO_PIECE

SQUARE_SIZE = 128
HALF_SQUARE_SIZE = SQUARE_SIZE // 2
BOARD_SIZE = 8 * SQUARE_SIZE
WARP_SIZE = BOARD_SIZE


def get_square(x, y):
    return f'{chr(x + 97)}{8 - y}'


def get_matrix(corners, shift):
    target = np.array([[BOARD_SIZE, BOARD_SIZE],
                       [0, BOARD_SIZE],
                       [0, 0],
                       [BOARD_SIZE, 0]], dtype=np.float32) + shift
    matrix = cv2.getPerspectiveTransform(corners, target)
    return matrix


def get_center_of_piece(bbox):
    x, y, w, h = bbox[1:].astype(float)
    center = [x + w / 2,
              y + h - w / 4]
    return center


def main(verbose=False):
    with open(os.path.join(LABEL_STUDIO_DIR, 'data/export/project-1-at-2023-02-11-17-55-378e444e.json')) as f:
        d = json.load(f)

    os.makedirs('data/classification')

    for label in tqdm(d):
        result = label['annotations'][0]['result']

        # Only keep images with key point annotations and at least one piece annotation
        n_keypoints = sum(['keypointlabels' in entry['value'] for entry in result])
        n_bboxes = sum(['rectanglelabels' in entry['value'] for entry in result])
        if n_keypoints != 4 or n_bboxes == 0:
            continue

        image_path = os.path.join(UPLOAD_DIR, *label['data']['img'].split('/')[-2:])
        if not os.path.isfile(image_path):
            image_path = f'label_studio/files/images/{os.path.basename(label["data"]["img"])}'
        image = np.array(Image.open(image_path).convert('RGB'))
        image_height, image_width = image.shape[:2]

        points = {}
        pieces = []
        piece_centers = []
        for entry in result:
            value = entry['value']
            if 'keypointlabels' in value:
                x = value['x'] * image_width / 100
                y = value['y'] * image_height / 100
                points[value['keypointlabels'][0]] = [x, y]
            else:
                piece = value['rectanglelabels'][0]
                pieces.append(piece)

                x = value['x'] * image_width / 100
                y = value['y'] * image_height / 100
                w = value['width'] * image_width / 100
                h = value['height'] * image_height / 100
                piece_center = [x + w / 2,
                                y + h - w / 4]
                piece_centers.append(piece_center)
        keypoints = np.array([points[s] for s in CORNERS], dtype=np.float32)
        piece_centers = np.array(piece_centers)

        matrix = get_matrix(keypoints, HALF_SQUARE_SIZE)
        inv_matrix = np.linalg.inv(matrix)
        warped_image = cv2.warpPerspective(image, matrix, (WARP_SIZE, WARP_SIZE))
        centers = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 1.0) * SQUARE_SIZE
        warped_centers = cv2.perspectiveTransform(np.expand_dims(centers, axis=0), inv_matrix)[0]
        dist_matrix = scipy.spatial.distance_matrix(piece_centers, warped_centers)

        matching = linear_sum_assignment(dist_matrix)
        square_to_piece = {get_square(j // 8, j % 8): pieces[i]
                           for i, j in zip(matching[0], matching[1])}

        for x in range(8):
            for y in range(8):
                left = int(x * SQUARE_SIZE)
                top = int(y * SQUARE_SIZE)
                crop = warped_image[top: top + 2 * SQUARE_SIZE, left: left + 2 * SQUARE_SIZE]

                square = get_square(x, y)
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                category = square_to_piece.get(square, 'empty')
                new_image_path = os.path.join(ROOT_DIR, 'data', 'classification',
                                              f'{image_id}_category={category}_square={square}.jpg')
                Image.fromarray(crop).save(new_image_path)

        if verbose:
            board = Board(fen=None)
            for square, piece in square_to_piece.items():
                board.set_piece_at(chess.SQUARES[chess.parse_square(square)], CLASS_TO_PIECE[piece])

            Image.fromarray(image).show()
            print(board)
            input()


if __name__ == '__main__':
    main()
