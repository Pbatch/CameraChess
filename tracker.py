import argparse
import os

import chess
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE, CLASSES, ABBR_MAP, SQUARE_TO_PIECE, PIECE_TO_CLASS
from camera_chess.detector import Detector
from camera_chess.utils import load_video_config
from camera_chess.video import Video


def _xy_to_square(x, y):
    return f'{chr(x + 97)}{8 - y}'


def _square_to_xy(square):
    x = ord(square[0]) - 97
    y = 8 - int(square[1])
    return x, y


def _init_state():
    state = np.zeros((8, 8, len(CLASSES)), dtype=np.float32)
    for square, piece in SQUARE_TO_PIECE.items():
        x, y = _square_to_xy(square)
        cls_idx = CLASSES.index(piece)
        state[x][y][cls_idx] = 1.0
    return state


def _update_state(state, arr, decay=0.5):
    state *= decay
    x = arr[:, 0].astype(np.int32)
    y = arr[:, 1].astype(np.int32)
    conf = arr[:, 2]
    cls = arr[:, 3].astype(np.int32)
    state[x, y, cls] += (1 - decay) * conf


class Creator:
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
    def _perspective_transform(src, matrix):
        homo_src = src.transpose(1, 0, 2)
        homo_src = np.concatenate([homo_src, np.ones((len(homo_src), 1, 1))], axis=2)
        warped_src = homo_src @ matrix.T
        warped_src /= warped_src[..., 2:]
        warped_src = warped_src[..., :2]
        warped_src = warped_src.transpose(1, 0, 2)
        return warped_src

    def _get_nearest_xy(self, center):
        idx = np.argmin(np.sum((center - self.square_centers) ** 2, axis=1))
        x = idx // 8
        y = idx % 8
        return x, y

    def _is_out_of_bounds(self, center):
        for i in range(4):
            v1 = self.boundary[i - 1] - self.boundary[i]
            v2 = center - self.boundary[i]
            cross_products = np.cross(v1, v2)
            if cross_products < 0:
                return True
        return False

    def _warp(self, src):
        target = np.array([[BOARD_SIZE, BOARD_SIZE],
                           [0, BOARD_SIZE],
                           [0, 0],
                           [BOARD_SIZE, 0]], dtype=np.float32)
        matrix = self._get_perspective_transform(target)
        inv_matrix = np.linalg.inv(matrix)
        warped_src = self._perspective_transform(np.expand_dims(src, axis=0), inv_matrix)[0]
        return warped_src

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

    def create_sequence(self, sequence_path):
        if os.path.isfile(sequence_path):
            return

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

                x, y = self._get_nearest_xy(center)
                new_pred = [i, x, y, pred[-2], pred[-1]]
                sequence.append(new_pred)

        np.save(sequence_path, sequence)

    def create_video(self, sequence_path):
        sequence = np.load(sequence_path)

        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('video.mp4', fourcc, self.video.target_fps, size)

        state = _init_state()
        idxs, breakpoints = np.unique(sequence[:, 0], return_index=True)
        arrs = np.split(sequence[:, 1:], breakpoints[1:])
        for idx, arr in tqdm(zip(idxs, arrs), total=len(idxs)):
            _update_state(state, arr)

            image = self._plot_state(state)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()


class Candidate:
    MOVE_PENALTY = 2.5

    def __init__(self):
        self.board = chess.Board()
        self.san_stack = []
        self.score = 0

    def calculate_hash(self):
        return hash(self.board.fen().split(' ', 1)[0])

    def calculate_score(self, state, move):
        probs = np.zeros(64, dtype=np.float32)
        piece_map = self.board.piece_map()
        for square in range(64):
            piece = piece_map.get(square, None)
            x, y = _square_to_xy(chess.square_name(square))
            if piece is None:
                p = 1 - max(state[x][y])
            else:
                cls = CLASSES.index(PIECE_TO_CLASS[piece])
                p = state[x][y][cls]
            probs[square] = p
        score = np.sum(np.log(probs + 0.01))
        if move is not None:
            score -= self.MOVE_PENALTY
        return score

    def push(self, move):
        if move is not None:
            self.san_stack.append(self.board.san(move))
            self.board.push(move)

    def pop(self, move):
        if move is not None:
            self.san_stack.pop()
            self.board.pop()

    def __str__(self):
        s = f'{self.score:.2f} '
        for i, move in enumerate(self.san_stack):
            if i % 2 == 0:
                s += f'{(i + 2) // 2}. {str(move)}'
            else:
                s += f' {str(move)} '
        return s


class Tracker:
    def _process_point(self, state, arr, new_candidates):
        _update_state(state, arr)

        old_candidates = new_candidates
        new_candidates = []
        seen = set()
        for candidate in old_candidates:
            moves = list(candidate.board.legal_moves) + [None]
            for move in moves:
                candidate.push(move)

                hash_ = candidate.calculate_hash()
                if hash_ not in seen:
                    seen.add(hash_)

                    score = candidate.calculate_score(state, move)

                    valid = False
                    if len(new_candidates) < 3:
                        valid = True
                    elif score > new_candidates[-1].score:
                        new_candidates.pop()
                        valid = True

                    if valid:
                        new_candidate = Candidate()
                        new_candidate.san_stack = candidate.san_stack[:]
                        new_candidate.board = candidate.board.copy(stack=False)
                        new_candidate.score = score
                        new_candidates.append(new_candidate)

                    new_candidates.sort(key=lambda x: x.score, reverse=True)
                candidate.pop(move)

        return new_candidates

    def process_sequence(self, sequence_path):
        sequence = np.load(sequence_path)

        logs = []
        state = _init_state()
        init_candidate = Candidate()
        init_candidate.score = init_candidate.calculate_score(state, None)
        new_candidates = [init_candidate]

        idxs, breakpoints = np.unique(sequence[:, 0], return_index=True)
        arrs = np.split(sequence[:, 1:], breakpoints[1:])
        for idx, arr in tqdm(zip(idxs, arrs), total=len(idxs)):
            new_candidates = self._process_point(state, arr, new_candidates)

            if idx % 10 == 0:
                for candidate in new_candidates:
                    logs.append(f'{idx} {candidate}')

        with open('logs.txt', 'w') as f:
            f.write('\n'.join(logs))


def main(dataset):
    sequence_path = f'{dataset.replace("/", "_")}_sequence.npy'

    # creator = Creator(dataset)
    # creator.create_sequence(sequence_path)

    tracker = Tracker()
    tracker.process_sequence(sequence_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
