import json
import os

import chess
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE, CLASSES, ABBR_MAP
from camera_chess.detector import Detector

from camera_chess.utils import load_video_config, update_state
from camera_chess.video import Video


class SequenceGenerator:
    PLOT_SIZE = 64

    def __init__(self, dataset):
        self.dataset = dataset
        self.video_config = load_video_config(self.dataset)
        self.video = Video(self.video_config, target_fps=8)

        self.centers, self.boundary = self._get_centers_and_boundary(self.video.new_keypoints)

        cmap = plt.get_cmap('Blues')
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        self.scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

    @staticmethod
    def _perspective_transform(src, matrix):
        if src.ndim == 2:
            src = np.expand_dims(src, axis=0)
        homo_src = src.transpose(1, 0, 2)
        homo_src = np.concatenate([homo_src, np.ones((len(homo_src), 1, 1))], axis=2)
        warped_src = homo_src @ matrix.T
        warped_src /= warped_src[..., 2:]
        warped_src = warped_src[..., :2]
        warped_src = warped_src.transpose(1, 0, 2)
        return warped_src

    @staticmethod
    def _get_perspective_transform(target, keypoints):
        A = np.zeros((8, 8), dtype=np.float32)
        B = np.zeros((8, 1), dtype=np.float32)

        for i in range(4):
            x, y = keypoints[i]
            u, v = target[i]
            A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            B[i * 2] = u
            B[i * 2 + 1] = v

        matrix = np.linalg.solve(A, B)
        matrix = np.append(matrix, 1.0)
        matrix = matrix.reshape((3, 3))
        return matrix

    def _get_centers_and_boundary(self, keypoints):
        target = np.array([[BOARD_SIZE, BOARD_SIZE],
                           [0, BOARD_SIZE],
                           [0, 0],
                           [BOARD_SIZE, 0]], dtype=np.float32)
        matrix = self._get_perspective_transform(target, keypoints)
        inv_matrix = np.linalg.inv(matrix)

        x = np.linspace(0.5, 7.5, num=8)
        y = np.linspace(7.5, 0.5, num=8)
        warped_centers = np.concatenate(np.meshgrid(x, y)).reshape(2, -1).T * SQUARE_SIZE
        centers = self._perspective_transform(warped_centers, inv_matrix)[0]

        warped_boundary = np.array([[-0.5, -0.5], [-0.5, 8.5], [8.5, 8.5], [8.5, -0.5]]) * SQUARE_SIZE
        boundary = self._perspective_transform(warped_boundary, inv_matrix)[0]

        return centers, boundary

    def _get_nearest_square(self, center):
        square = np.argmin(np.sum((center - self.centers) ** 2, axis=1))
        return square

    def _is_out_of_bounds(self, center):
        for i in range(4):
            a = self.boundary[i - 1][0] - self.boundary[i][0]
            b = self.boundary[i - 1][1] - self.boundary[i][1]
            c = center[0] - self.boundary[i][0]
            d = center[1] - self.boundary[i][1]
            cross_product = (a * d) - (b * c)
            if cross_product < 0:
                return True
        return False

    def _plot_state(self, state, from_square, to_square, thr=0.7):
        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        image = Image.new('RGB', size)
        d = ImageDraw.Draw(image)
        max_idx = np.argmax(state, axis=1)
        for square in range(64):
            cls = max_idx[square]
            score = state[square][cls]

            x = square % 8
            y = 7 - (square // 8)
            bbox = [self.PLOT_SIZE * i for i in [x, y, (x + 1), (y + 1)]]
            fill = colors.rgb2hex(self.scalar_map.to_rgba((score - thr) / (1 - thr)))

            if square in {from_square, to_square}:
                outline = "yellow"
                width = 5
            else:
                outline = "black"
                width = 1
            d.rectangle(bbox, fill=fill, outline=outline, width=width)

            if score > thr:
                d.text([self.PLOT_SIZE * (x + 0.45), self.PLOT_SIZE * (y + 0.45)],
                       ABBR_MAP[CLASSES[cls]],
                       fill='black')
        return image

    def create_sequence(self, sequence_path, boxes_path):
        if os.path.isfile(sequence_path) and os.path.isfile(boxes_path):
            sequence = np.load(sequence_path)
            boxes = np.load(boxes_path)
            return sequence, boxes

        detector = Detector(model_path='models/480L.pt',
                            device='cuda')
        sequence = np.zeros((len(self.video), 64, len(CLASSES)))
        boxes = []
        for i, (image, frame) in tqdm(enumerate(self.video), desc='Frame'):
            preds = detector.run(np.expand_dims(image, axis=0))[0]

            cx = (preds[:, 0] + preds[:, 2]) / 2
            cy = preds[:, 3] - ((preds[:, 2] - preds[:, 0]) / 3)
            box_centers = np.vstack((cx, cy)).T

            dist = np.sum(np.square(np.expand_dims(box_centers, 1) - np.expand_dims(self.centers, 0)), axis=2)
            squares = np.argmin(dist, axis=1)
            for square, box_center, pred in zip(squares, box_centers, preds):
                oob = self._is_out_of_bounds(box_center)
                if oob:
                    continue

                for k in range(len(CLASSES)):
                    sequence[i][square][k] = max(sequence[i][square][k], pred[4 + k])

                conf = max(pred[4:])
                boxes.append([i, square, *pred[:4], conf])

        sequence = np.asarray(sequence)
        np.save(sequence_path, sequence)

        boxes = np.asarray(boxes)
        boxes = boxes[(-boxes[:, -1]).argsort()]
        np.save(boxes_path, boxes)

        return sequence, boxes

    def create_video(self, sequence_path, video_path, logs_path=None):
        sequence = np.load(sequence_path)

        if logs_path is not None:
            with open(logs_path, 'rb') as f:
                logs = json.load(f)
        else:
            logs = {}

        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(video_path, fourcc, self.video.target_fps, size)

        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        from_square = None
        to_square = None
        board = chess.Board()
        for i in range(len(sequence)):
            update_state(state, sequence[i])

            if i in logs:
                d = logs[i]
                uci_move = board.parse_san(d['moves'].split()[0])
                board.push(uci_move)
                from_square = uci_move.from_square
                to_square = uci_move.to_square

            image = self._plot_state(state, from_square, to_square)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()
