import argparse
import json
import os
import sys

import chess
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from PIL import Image, ImageDraw
from icecream import ic
from matplotlib import pyplot as plt
from tqdm import tqdm

from camera_chess.constants import CLASSES, ABBR_MAP, PIECE_TO_CLASS
from camera_chess.detector import Detector
from camera_chess.utils import load_video_config
from camera_chess.video import Video

sys.path.append('../CameraChessWeb')
from aws.tracker.lambda_function import get_centers_and_boundary


def _update_state(state, arr, decay=0.5):
    state *= decay
    square = arr[:, 0].astype(np.int32)
    conf = arr[:, 1]
    cls = arr[:, 2].astype(np.int32)
    state[square, cls] += (1 - decay) * conf


class Creator:
    PLOT_SIZE = 64

    def __init__(self, dataset):
        self.dataset = dataset
        self.video_config = load_video_config(self.dataset)
        self.video = Video(self.video_config, target_fps=8)

        self.centers, self.boundary = get_centers_and_boundary(self.video.new_keypoints)

        cmap = plt.get_cmap('Blues')
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        self.scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

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

                square = self._get_nearest_square(center)
                new_pred = [i, square, pred[-2], pred[-1]]
                sequence.append(new_pred)

        np.save(sequence_path, sequence)

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
        for idx, arr in tqdm(zip(idxs, arrs), total=len(idxs)):
            _update_state(state, arr)

            idx = str(idx)
            if idx in logs:
                candidates = logs[idx]
                best_candidate = max(candidates, key=lambda x: x["score"])
                board = chess.Board()
                for san_move in best_candidate["san"]:
                    uci_move = board.parse_san(san_move)
                    board.push(uci_move)
                    from_square = uci_move.from_square
                    to_square = uci_move.to_square

            image = self._plot_state(state, from_square, to_square)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()


class Tracker:
    def _get_from_and_to_squares(self, move, board):
        from_squares = []
        to_squares = []
        if board.is_castling(move) or board.is_en_passant(move):
            next_board = board.copy(stack=False)
            next_board.push(move)
            for square in range(64):
                board_piece = PIECE_TO_CLASS.get(board.piece_at(square), None)
                next_board_piece = PIECE_TO_CLASS.get(next_board.piece_at(square), None)
                if board_piece == next_board_piece:
                    continue

                if next_board_piece is None:
                    from_squares.append(square)

                if board_piece is None:
                    to_squares.append(square)
        else:
            from_squares.append(move.from_square)
            to_squares.append(move.to_square)
        return from_squares, to_squares

    def _get_squares(self, move_1, board, move_2=None):
        from_squares_1, to_squares_1 = self._get_from_and_to_squares(move_1, board)
        if move_2 is None:
            return from_squares_1, to_squares_1

        board.push(move_1)
        from_squares_2, to_squares_2 = self._get_from_and_to_squares(move_2, board)
        board.pop()

        squares_2 = from_squares_2 + to_squares_2
        from_squares = list(set(from_squares_1) - set(squares_2)) + from_squares_2
        to_squares = list(set(to_squares_1) - set(squares_2)) + to_squares_2

        return from_squares, to_squares

    def _calculate_score(self, state, move_1, board, move_2=None):
        from_squares, to_squares = self._get_squares(move_1, board, move_2)

        score = 0
        for square in from_squares:
            score += 1 - max(state[square]) - 0.7

        board.push(move_1)
        if move_2 is not None:
            board.push(move_2)

        for square in to_squares:
            cls = CLASSES.index(PIECE_TO_CLASS[board.piece_at(square)])
            score += state[square][cls] - 0.7
        board.pop()
        if move_2 is not None:
            board.pop()

        return score

    def _process_point(self, state, arr, board, moves):
        _update_state(state, arr)

        if not len(moves):
            for move_1 in board.legal_moves:
                board.push(move_1)
                for move_2 in board.legal_moves:
                    moves.append([move_1, move_2])
                board.pop()

        best_moves = []
        best_score = -float('inf')
        for move_1, move_2 in moves:
            board.push(move_1)
            move_2_score = self._calculate_score(state, move_2, board)
            board.pop()

            move_1_score = self._calculate_score(state, move_1, board)
            joint_score = self._calculate_score(state, move_1, board, move_2)

            if move_2_score < 0 or joint_score < 0:
                continue

            ic(move_1, move_2, move_1_score, move_2_score, joint_score)

            if joint_score > best_score:
                best_score = joint_score
                best_moves = [move_1, move_2]
                ic(best_score, best_moves)

        return best_moves, best_score

    def process_sequence(self, sequence_path, logs_path):
        sequence = np.load(sequence_path)

        logs = {}
        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        board = chess.Board()
        moves = []

        idxs, breakpoints = np.unique(sequence[:, 0], return_index=True)
        arrs = np.split(sequence[:, 1:], breakpoints[1:])
        for idx, arr in tqdm(zip(idxs, arrs), total=len(idxs)):
            best_moves, best_score = self._process_point(state, arr, board, moves)

            if best_score > 0:
                board.push(best_moves[0])
                logs[idx] = {"best_moves": ' '.join([move.uci() for move in best_moves]),
                             "best_score": best_score,
                             "stack": ' '.join([move.uci() for move in board.move_stack])}
                moves = []

        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2)


def main(dataset):
    sequence_path = f'{dataset.replace("/", "_")}_sequence.npy'
    logs_path = f'{os.path.splitext(os.path.basename(sequence_path))[0]}_logs.json'
    video_path = f'{os.path.splitext(os.path.basename(sequence_path))[0]}_video.mp4'

    creator = Creator(dataset)
    creator.create_sequence(sequence_path)

    tracker = Tracker()
    tracker.process_sequence(sequence_path, logs_path)

    # creator.create_video(sequence_path, video_path, logs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
