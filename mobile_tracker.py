import argparse
import os

import chess
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from PIL import Image, ImageDraw
from icecream import ic
from matplotlib import pyplot as plt
from tqdm import tqdm

from camera_chess.classifier_predict import MobileDetector
from camera_chess.constants import CLASSES, ABBR_MAP, PIECE_TO_CLASS
from camera_chess.utils import load_video_config
from camera_chess.video import Video


def _square_to_xy(square):
    x = ord(square[0]) - 97
    y = 8 - int(square[1])
    return x, y


def _xy_to_square(x, y):
    # 0, 0 -> h1
    # 0, 1 -> h2
    # 1, 0 -> g1
    return f'{chr((7 - x) + 97)}{y + 1}'


def _init_state():
    state = np.zeros((64, len(CLASSES)), dtype=np.float32)
    return state


def _update_state(state, update, decay=0.5):
    return decay * state + (1 - decay) * update


class Creator:
    PLOT_SIZE = 64

    def __init__(self, dataset):
        self.dataset = dataset
        self.video_config = load_video_config(self.dataset)
        self.video = Video(self.video_config, target_fps=4)

        cmap = plt.get_cmap('Blues')
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        self.scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

    def _plot_state(self, state):
        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        image = Image.new('RGB', size)
        d = ImageDraw.Draw(image)
        cls_idxs = np.argmax(state, axis=1)
        for i in range(64):
            if cls_idxs[i] == 12:
                continue

            x = 7 - (i // 8)
            y = 7 - (i % 8)

            bbox = [self.PLOT_SIZE * i for i in [x, y, (x + 1), (y + 1)]]
            fill = colors.rgb2hex(self.scalar_map.to_rgba(state[i][cls_idxs[i]]))
            d.rectangle(bbox, fill=fill, outline="black")
            d.text([self.PLOT_SIZE * (x + 0.45), self.PLOT_SIZE * (y + 0.45)],
                   ABBR_MAP[CLASSES[cls_idxs[i]]],
                   fill='black')
        return image

    def create_sequence(self, sequence_path):
        if os.path.isfile(sequence_path):
            return

        detector = MobileDetector(checkpoint_path='ChessClassifier/cq4zp3xk/checkpoints/epoch=3-step=20040.ckpt',
                                  device='cuda')
        sequence = []
        for i, (image, frame) in tqdm(enumerate(self.video), desc='Frame'):
            preds = detector.run(image, self.video.new_keypoints)
            sequence.append(preds.tolist())

        np.save(sequence_path, sequence)

    def create_video(self, sequence_path):
        sequence = np.load(sequence_path)

        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('video.mp4', fourcc, self.video.target_fps, size)

        state = _init_state()
        for update in tqdm(sequence):
            state = _update_state(state, update)

            image = self._plot_state(state)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()


class Candidate:
    MOVE_PENALTY = 3.0

    def __init__(self):
        self.board = chess.Board()
        self.san_stack = []
        self.score = 0

    def calculate_hash(self):
        return hash(self.board.fen().split(' ', 1)[0])

    def calculate_score(self, state, move):
        probs = np.zeros(64, dtype=np.float32)
        piece_map = self.board.piece_map()
        for i in range(64):
            x = i // 8
            y = i % 8
            square = _xy_to_square(x, y)

            piece = piece_map.get(chess.parse_square(square), 'empty')
            cls = CLASSES.index(PIECE_TO_CLASS[piece])
            p = state[i][cls]
            probs[i] = p
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
        state = _update_state(state, arr)

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

        return state, new_candidates

    def process_sequence(self, sequence_path):
        sequence = np.load(sequence_path)

        logs = []
        state = _init_state()
        init_candidate = Candidate()
        init_candidate.score = init_candidate.calculate_score(state, None)
        new_candidates = [init_candidate]

        for i, update in enumerate(sequence):
            state, new_candidates = self._process_point(state, update, new_candidates)

            if i % 10 == 0:
                for candidate in new_candidates:
                    logs.append(f'{i} {candidate}')

        with open('logs.txt', 'w') as f:
            f.write('\n'.join(logs))


def main(dataset):
    sequence_path = f'{dataset.replace("/", "_")}_sequence.npy'

    creator = Creator(dataset)
    creator.create_sequence(sequence_path)
    # creator.create_video(sequence_path)

    tracker = Tracker()
    tracker.process_sequence(sequence_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
