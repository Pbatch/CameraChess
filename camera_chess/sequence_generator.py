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
            v1 = self.boundary[i - 1] - self.boundary[i]
            v2 = center - self.boundary[i]
            cross_products = np.cross(v1, v2)
            if cross_products < 0:
                return True
        return False

    def _plot_state(self, state, from_square, to_square, thr=0.2):
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
            fill = colors.rgb2hex(self.scalar_map.to_rgba(score))

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

    def create_sequence(self, sequence_path):
        if os.path.isfile(sequence_path):
            sequence = np.load(sequence_path)
            return sequence

        detector = Detector(model_path='models/480L.pt',
                            device='cuda')
        sequence = []
        for i, (image, frame) in tqdm(enumerate(self.video), desc='Frame'):
            preds = detector.run(np.expand_dims(image, axis=0))[0]

            for pred in preds:
                center = np.array([(pred[0] + pred[2]) / 2,
                                   pred[3] - ((pred[2] - pred[0]) / 4)])
                oob = self._is_out_of_bounds(center)
                if oob:
                    continue

                square = self._get_nearest_square(center)
                new_pred = [i, square, *pred]
                sequence.append(new_pred)

        sequence = np.asarray(sequence)
        np.save(sequence_path, sequence)

        return sequence

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
        idxs, breakpoints = np.unique(sequence[:, 0], return_index=True)
        arrs = np.split(sequence[:, 1:], breakpoints[1:])
        from_square = None
        to_square = None
        board = chess.Board()
        for idx, arr in tqdm(zip(idxs, arrs), total=len(idxs)):
            update_state(state, arr)

            idx = str(idx)
            if idx in logs:
                d = logs[idx]
                uci_move = board.parse_san(d['moves'].split()[0])
                board.push(uci_move)
                from_square = uci_move.from_square
                to_square = uci_move.to_square

            image = self._plot_state(state, from_square, to_square)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()
