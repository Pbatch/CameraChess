import argparse
import json
import os

import chess
import numpy as np
from icecream import ic
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR
from camera_chess.move_data_generator import MoveDataGenerator
from camera_chess.sequence_generator import SequenceGenerator
from camera_chess.utils import update_state


class Tracker:
    def __init__(self, from_thr=0.6, to_thr=0.6, decay=0.5):
        self.from_thr = from_thr
        self.to_thr = to_thr
        self.decay = decay

    def _calculate_score(self, state, move):
        score = 0
        for square in move['from']:
            score += 1 - max(state[square]) - self.from_thr

        for square, target in zip(move['to'], move['targets']):
            score += state[square][target] - self.to_thr

        return score

    def _process_point(self, state, move_data, possible_moves):
        best_move = None
        best_moves = None
        best_score_1 = -float('inf')
        best_score_2 = -float('inf')
        best_joint_score = -float('inf')
        for d in move_data:
            score_1 = self._calculate_score(state, d["move1"])
            if score_1 > 0:
                possible_moves.add(d["move1"]["san"])
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

        return best_score_1, best_score_2, best_joint_score, best_move, best_moves, possible_moves

    def process_sequence(self, sequence_path, logs_path):
        if os.path.isfile(logs_path):
            with open(logs_path, 'r') as f:
                logs = json.load(f)
            return logs

        sequence = np.load(sequence_path)

        logs = {}
        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        board = chess.Board()
        pgn = ''
        move_count = 0
        move_data_generator = MoveDataGenerator()
        move_data = move_data_generator.run(board)
        possible_moves = set()

        for i in tqdm(range(len(sequence))):
            update_state(state, sequence[i], self.decay)
            best_score_1, best_score_2, best_joint_score, best_move, best_moves, possible_moves = \
                self._process_point(state, move_data, possible_moves)

            if best_joint_score > 0 and best_moves["san"][0] in possible_moves:
                ic(possible_moves)
                san_move = best_moves["san"][0]
                board.push(board.parse_san(san_move))
                if move_count % 2 == 0:
                    pgn += f'{(move_count + 2) // 2}. '
                pgn += f'{san_move} '
                move_count += 1
                logs[i] = {"moves": ' '.join(best_moves["san"]),
                           "score": best_joint_score,
                           "pgn": pgn}
                ic(best_moves['san'][0])
                ic(best_joint_score)
                ic(pgn)
                move_data = move_data_generator.run(board)
                possible_moves = set()

        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2)

        return logs


def main(dataset):
    sequence_path = os.path.join(DATA_DIR, dataset, 'sequence.npy')
    boxes_path = os.path.join(DATA_DIR, dataset, 'boxes.npy')
    logs_path = os.path.join(DATA_DIR, dataset, 'logs.json')
    video_path = os.path.join(DATA_DIR, dataset, 'video.mp4')

    sequence_generator = SequenceGenerator(dataset)
    sequence_generator.create_sequence(sequence_path, boxes_path)

    tracker = Tracker()
    tracker.process_sequence(sequence_path, logs_path)

    # sequence_generator.create_video(sequence_path, video_path, logs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
