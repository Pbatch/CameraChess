import os
from glob import glob

import os
from glob import glob

import chess
import cv2
import numpy as np
import scipy
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO

from camera_chess.constants import CLASSES, CLASS_TO_PIECE
from camera_chess.make_classification_data import get_matrix, get_square, SQUARE_SIZE


def predict(image_paths):
    model = YOLO(model='data/best.pt')
    results = model.predict(image_paths, save=True, save_txt=True, save_conf=True, half=True, stream=True)
    for _ in tqdm(results):
        pass


def load_preds(save_dir):
    preds = []
    for pred_path in sorted(glob(os.path.join(save_dir, 'labels', '*.txt')),
                            key=lambda x: int(os.path.basename(x).replace('image', '').replace('.txt', ''))):
        with open(pred_path, 'r') as f:
            pred = [[float(i) for i in line.strip().split()] for line in f.readlines()]
        preds.append(pred)
    return preds


def main():
    image_paths = sorted(list(glob('data/hikaru/crops/*')))
    # label_paths = sorted(list(glob('data/google/labels/*')))
    predict(image_paths)

    save_dir = 'runs/detect/predict26'
    preds = load_preds(save_dir)

    for image_path, pred in zip(image_paths, preds):
        image = Image.open(image_path)
        width, height = image.width, image.height

        # with open(label_path) as f:
        #     label = json.load(f)
        # keypoints = np.array([label['keypoints'][s] for s in CORNERS], dtype=np.float32)
        keypoints = np.array([[177, 476], [16, 171], [582, 71], [885, 318]], dtype=np.float32)
        # keypoints[:, 0] *= width
        # keypoints[:, 1] *= height

        pieces = []
        piece_centers = []
        for class_id, *bbox, conf in pred:
            piece = CLASSES[int(class_id)]
            pieces.append(piece)

            piece_center = [bbox[0] * width,
                            (bbox[1] + bbox[2] / 4) * height]
            piece_centers.append(piece_center)

        matrix = get_matrix(keypoints, 0)
        inv_matrix = np.linalg.inv(matrix)
        centers = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        warped_centers = cv2.perspectiveTransform(np.expand_dims(centers, axis=0), inv_matrix)[0]
        dist_matrix = scipy.spatial.distance_matrix(piece_centers, warped_centers)

        matching = linear_sum_assignment(dist_matrix)
        square_to_piece = {get_square(j // 8, j % 8): pieces[i]
                           for i, j in zip(matching[0], matching[1])}
        board = chess.Board(fen=None)
        for square, piece in square_to_piece.items():
            board.set_piece_at(chess.SQUARES[chess.parse_square(square)], CLASS_TO_PIECE[piece])

        print(board)
        image.show()
        input()


if __name__ == '__main__':
    main()
