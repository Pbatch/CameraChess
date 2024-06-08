import argparse
import io
import json
import os

import chess
import numpy as np
from loguru import logger
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR
from camera_chess.move_data_generator import MoveDataGenerator
from camera_chess.sequence_generator import SequenceGenerator
from camera_chess.utils import update_state, load_video_config


class Tracker:
    def __init__(self, dataset, model_basename, from_thr=0.6, to_thr=0.6, possible_thr=0.0, decay=0.5):
        self.dataset = dataset
        self.model_basename = model_basename
        self.from_thr = from_thr
        self.to_thr = to_thr
        self.possible_thr = possible_thr
        self.decay = decay

        self.logs_path = os.path.join(DATA_DIR, self.dataset, self.model_basename.split('.')[0], 'logs.json')

    @staticmethod
    def _update(board, san_move, san_moves, pgn, score, logs, key):
        move_count = len(board.move_stack)
        board.push(board.parse_san(san_move))
        if move_count % 2 == 0:
            pgn += f'{(move_count + 2) // 2}. '
        pgn += f'{san_move} '
        logs[str(key)] = {"moves": san_moves,
                          "score": score,
                          "pgn": pgn}
        # tqdm.write(str(logs[str(key)]))
        return board, pgn

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
            if score_1 > self.possible_thr:
                old_score = possible_moves.get(d["move1"]["san"], self.possible_thr)
                possible_moves[d["move1"]["san"]] = max(old_score, score_1)
            if score_1 > best_score_1:
                best_score_1 = score_1
                best_move = d["move1"]

            if "move2" not in d or d["move1"]["san"] not in possible_moves:
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

    def process_sequence(self, sequence_path, force=False):
        if os.path.isfile(self.logs_path) and not force:
            with open(self.logs_path, 'r') as f:
                logs = json.load(f)
            return logs

        sequence = np.load(sequence_path)

        logs = {}
        video_config = load_video_config(self.dataset)
        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        board = chess.Board(video_config.fen)
        pgn = ''
        move_data_generator = MoveDataGenerator()
        move_data = move_data_generator.run(board)
        possible_moves = {}

        for i in tqdm(range(len(sequence)), desc='Processing sequence'):
            update_state(state, sequence[i], self.decay)
            best_score_1, best_score_2, best_joint_score, best_move, best_moves, possible_moves = \
                self._process_point(state, move_data, possible_moves)

            push_move = best_joint_score > 0 and best_moves["san"][0] in possible_moves
            if push_move:
                board, pgn = self._update(board=board,
                                          san_move=best_moves["san"][0],
                                          san_moves=' '.join(best_moves["san"]),
                                          pgn=pgn,
                                          score=best_joint_score,
                                          logs=logs,
                                          key=i)
                move_data = move_data_generator.run(board)
                possible_moves.clear()

            if best_score_1 > 0 and not push_move and i == len(sequence) - 1:
                self._update(board=board,
                             san_move=best_move["san"],
                             san_moves=best_move["san"],
                             pgn=pgn,
                             score=best_score_1,
                             logs=logs,
                             key=i)

        with open(self.logs_path, 'w') as f:
            json.dump(logs, f, indent=2)

        pred_pgn = logs[max(logs.keys(), key=lambda x: int(x))]['pgn']
        pred_moves = [str(move) for move in chess.pgn.read_game(io.StringIO(pred_pgn)).mainline_moves()]

        gt_moves = video_config.moves

        score = 0
        pred_fail = None
        gt_fail = None
        halfmove_fail = -1
        board = chess.Board()
        for i in range(len(gt_moves)):
            pred_move = pred_moves[i] if i < len(pred_moves) else None
            gt_move = gt_moves[i]
            if pred_move == gt_move:
                score += 1
                board.push(board.parse_uci(gt_move))
            else:
                pred_fail = board.san(board.parse_uci(pred_move)) if pred_move is not None else ""
                gt_fail = board.san(board.parse_uci(gt_move))
                halfmove_fail = i
                break
        score = round(100 * score / len(gt_moves))
        info = {'score': score,
                'halfmoves': len(gt_moves),
                'pred_fail': pred_fail,
                'gt_fail': gt_fail,
                'halfmove_fail': halfmove_fail
                }
        logger.info(info)

        return logs


def main(dataset, model_basename, force_sequence, force_tracker, debug):
    sequence_generator = SequenceGenerator(dataset, model_basename)
    sequence_generator.create_sequence(force=force_sequence, debug=debug)

    tracker = Tracker(dataset, model_basename=sequence_generator.model_basename)
    tracker.process_sequence(sequence_generator.sequence_path, force=force_tracker)

    if debug:
        sequence_generator.create_video(tracker.logs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--model_basename', '-m', type=str, default="480S_v10_pieces_480x288.onnx")
    parser.add_argument('--force_sequence', '-fs', action="store_true")
    parser.add_argument('--force_tracker', '-ft', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.dataset, args.model_basename, args.force_sequence, args.force_tracker, args.debug)
