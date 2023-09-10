import argparse
import os
from collections import defaultdict
from copy import deepcopy

import chess
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE, CLASSES, ABBR_MAP, SQUARE_TO_PIECE, PIECE_TO_CLASS
from camera_chess.detector import Detector
from camera_chess.utils import load_video_config
from camera_chess.video import Video


class Board:
    MOVE_REWARD = -np.log(0.5)

    def __init__(self):
        self.board = chess.Board()
        self.san_stack = []
        self.idxs = []
        self.hash = hash(self.board.fen().split(' ', 1)[0])
        self.legal_moves = list(self.board.legal_moves)
        self.score = 0

    def push(self, move, score, idx):
        self.san_stack.append(self.board.san(move))
        self.idxs.append(idx)
        self.board.push(move)

        self.hash = hash(self.board.fen().split(' ', 1)[0])
        self.legal_moves = list(self.board.legal_moves)
        self.score += np.log(score) + self.MOVE_REWARD

    def __str__(self):
        s = f'{self.score:.2f} '
        for i, move in enumerate(self.san_stack):
            if i % 2 == 0:
                s += f'{(i + 2) // 2}. {str(move)}'
            else:
                s += f' {str(move)} '
        return s


class Tracker:
    PLOT_SIZE = 64

    def __init__(self, dataset):
        self.dataset = dataset

        self.video_config = load_video_config(self.dataset)
        self.video = Video(self.video_config, target_fps=8)

        self.square_centers = self._warp((np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE)
        self.boundary = self._warp(np.array([[-0.5, -0.5], [-0.5, 8.5], [8.5, 8.5], [8.5, -0.5]]) * SQUARE_SIZE)

        cmap = plt.get_cmap('Blues')
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        self.scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

    @staticmethod
    def _xy_to_square(x, y):
        return f'{chr(x + 97)}{8 - y}'

    @staticmethod
    def _square_to_xy(square):
        x = ord(square[0]) - 97
        y = 8 - int(square[1])
        return x, y

    @staticmethod
    def _perspective_transform(src, matrix):
        homo_src = src.transpose(1, 0, 2)
        homo_src = np.concatenate([homo_src, np.ones((len(homo_src), 1, 1))], axis=2)
        warped_src = homo_src @ matrix.T
        warped_src /= warped_src[..., 2:]
        warped_src = warped_src[..., :2]
        warped_src = warped_src.transpose(1, 0, 2)
        return warped_src

    def _update_state(self, state, df, decay=0.5):
        state *= decay
        for square, conf, cls in zip(*[df[s] for s in ['square', 'conf', 'cls']]):
            x, y = self._square_to_xy(square)
            state[x][y][cls] += (1 - decay) * conf

    def _init_state(self):
        state = np.zeros((8, 8, len(CLASSES)), dtype=np.float32)
        for square, piece in SQUARE_TO_PIECE.items():
            x, y = self._square_to_xy(square)
            cls_idx = CLASSES.index(piece)
            state[x][y][cls_idx] = 1.0
        return state

    def _get_perspective_transform(self, target):
        A = np.zeros((8, 8), dtype=np.float32)
        B = np.zeros((8, 1), dtype=np.float32)

        for i in range(4):
            x, y = self.video.new_keypoints[i]
            u, v = target[i]
            A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            B[i * 2] = u
            B[i * 2 + 1] = v

        matrix = np.linalg.solve(A, B)
        matrix = np.append(matrix, 1.0)
        matrix = matrix.reshape((3, 3))
        return matrix

    def _warp(self, src):
        target = np.array([[BOARD_SIZE, BOARD_SIZE],
                           [0, BOARD_SIZE],
                           [0, 0],
                           [BOARD_SIZE, 0]], dtype=np.float32)
        matrix = self._get_perspective_transform(target)
        inv_matrix = np.linalg.inv(matrix)
        warped_src = self._perspective_transform(np.expand_dims(src, axis=0), inv_matrix)[0]
        return warped_src

    def _get_nearest_square(self, center):
        idx = np.argmin(np.sum((center - self.square_centers) ** 2, axis=1))
        x = idx // 8
        y = idx % 8
        return self._xy_to_square(x, y)

    def _is_out_of_bounds(self, center):
        for i in range(4):
            v1 = self.boundary[i - 1] - self.boundary[i]
            v2 = center - self.boundary[i]
            cross_products = np.cross(v1, v2)
            if cross_products < 0:
                return True
        return False

    def _create_sequence(self, sequence_path):
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
                new_pred = [i, *pred[:5], int(pred[5]), square]
                sequence.append(new_pred)

        df = pd.DataFrame(sequence, columns=['idx', 'l', 't', 'r', 'b', 'conf', 'cls', 'square'])
        df.to_csv(sequence_path, index=False)

    def _plot_state(self, state, thr=0.2):
        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        image = Image.new('RGB', size)
        d = ImageDraw.Draw(image)
        max_idx = np.argmax(state, axis=2)
        for x in range(8):
            for y in range(8):
                cls = max_idx[x][y]
                score = state[x][y][cls]

                bbox = [self.PLOT_SIZE * i for i in [x, y, (x + 1), (y + 1)]]
                fill = colors.rgb2hex(self.scalar_map.to_rgba(score))
                d.rectangle(bbox, fill=fill, outline="black")

                if score > thr:
                    d.text([self.PLOT_SIZE * (x + 0.45), self.PLOT_SIZE * (y + 0.45)],
                           ABBR_MAP[CLASSES[cls]],
                           fill='black')
        return image

    def _create_video(self, sequence_path):
        df = pd.read_csv(sequence_path)

        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('video.mp4', fourcc, self.video.target_fps, size)

        state = self._init_state()
        for _, idx_df in tqdm(df.groupby('idx')):
            self._update_state(state, idx_df)

            image = self._plot_state(state)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()

    def _parse_sequence(self, sequence_path):
        df = pd.read_csv(sequence_path)

        state = self._init_state()
        new_boards = [Board()]
        for idx, idx_df in tqdm(df.groupby('idx')):
            self._update_state(state, idx_df)

            old_boards = new_boards
            hash_to_candidates = defaultdict(list)
            for board in old_boards:
                hash_to_candidates[board.hash].append(board)

                best_move = None
                best_score = 0
                for move in board.legal_moves:
                    piece = PIECE_TO_CLASS[board.board.piece_at(move.from_square)]
                    cls = CLASSES.index(piece)

                    from_square = chess.square_name(move.from_square)
                    from_x, from_y = self._square_to_xy(from_square)
                    from_score = state[from_x][from_y][cls]

                    to_square = chess.square_name(move.to_square)
                    to_x, to_y = self._square_to_xy(to_square)
                    to_score = state[to_x][to_y][cls]

                    score = to_score * (1 - from_score)
                    if score > best_score:
                        best_score = score
                        best_move = move

                if best_move is None:
                    continue

                if Board.MOVE_REWARD + np.log(best_score) < 0:
                    continue

                new_board = deepcopy(board)
                new_board.push(best_move, best_score, idx)
                hash_to_candidates[new_board.hash].append(new_board)

            candidates = []
            for v in hash_to_candidates.values():
                candidate = min(v, key=lambda x: x.idxs)
                candidate.score = max(v, key=lambda x: x.score).score
                candidates.append(candidate)
            new_boards = sorted(candidates, key=lambda x: x.score, reverse=True)[:5]

        print(new_boards[0])

    def run(self):
        sequence_path = f'{self.dataset.replace("/", "_")}_sequence.npy'
        if not os.path.isfile(sequence_path):
            self._create_sequence(sequence_path)

        # self._create_video(sequence_path)
        self._parse_sequence(sequence_path)


def main(dataset):
    tracker = Tracker(dataset)
    tracker.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
