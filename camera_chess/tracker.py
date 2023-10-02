import argparse
import json
import os

import chess
import numpy as np
from tqdm import tqdm

from camera_chess.constants import CLASSES
from camera_chess.move_data_generator import MoveDataGenerator
from camera_chess.sequence_generator import SequenceGenerator
from camera_chess.utils import update_state


class Tracker:
    def _calculate_score(self, state, move):
        score = 0
        for square in move['from']:
            score += 1 - max(state[square]) - 0.7

        for square, target in zip(move['to'], move['targets']):
            score += state[square][target] - 0.7

        return score

    def _process_point(self, state, move_data):
        best_move = None
        best_moves = None
        best_score_1 = -float('inf')
        best_score_2 = -float('inf')
        best_joint_score = -float('inf')
        for d in move_data:
            score_1 = self._calculate_score(state, d["move1"])
            if score_1 > best_score_1:
                best_score_1 = score_1
                best_move = d["move1"]

            if "move2" not in d:
                continue

            score_2 = self._calculate_score(state, d["move2"])
            if score_2 < 0:
                continue
            elif score_2 > best_score_2:
                best_score_2 = score_2

            joint_score = self._calculate_score(state, d["moves"])
            if joint_score > best_joint_score:
                best_joint_score = joint_score
                best_moves = d["moves"]

        return best_score_1, best_score_2, best_joint_score, best_move, best_moves

    def process_sequence(self, sequence_path, logs_path):
        if os.path.isfile(logs_path):
            with open(logs_path, 'r') as f:
                logs = json.load(f)
            return logs

        sequence = np.load(sequence_path)

        logs = {}
        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        board = chess.Board()
        move_data_generator = MoveDataGenerator()
        move_data = move_data_generator.run(board)

        idxs, breakpoints = np.unique(sequence[:, 0], return_index=True)
        arrs = np.split(sequence[:, 1:], breakpoints[1:])
        for idx, arr in tqdm(zip(idxs, arrs), total=len(idxs)):
            update_state(state, arr)
            best_score_1, best_score_2, best_joint_score, best_move, best_moves = self._process_point(state, move_data)

            if best_joint_score > 0:
                board.push(board.parse_san(best_moves["san"][0]))
                logs[idx] = {"moves": ' '.join(best_moves["san"]),
                             "score": best_joint_score,
                             "stack": ' '.join([move.uci() for move in board.move_stack])}
                move_data = move_data_generator.run(board)

        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2)

        return logs


def main(dataset):
    sequence_path = f'{dataset.replace("/", "_")}_sequence.npy'
    logs_path = f'{os.path.splitext(os.path.basename(sequence_path))[0]}_logs.json'
    video_path = f'{os.path.splitext(os.path.basename(sequence_path))[0]}_video.mp4'

    sequence_generator = SequenceGenerator(dataset)
    # creator.create_sequence(sequence_path)

    tracker = Tracker()
    tracker.process_sequence(sequence_path, logs_path)

    sequence_generator.create_video(sequence_path, video_path, logs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
