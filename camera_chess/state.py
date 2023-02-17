import io
import random

import cairosvg
import chess
import chess.pgn
import chess.svg
from PIL import Image
from icecream import ic

from camera_chess.constants import PIECE_TO_CLASS


class Action:
    def __init__(self, move, square_to_pred, piece):
        self.move = move
        self.square_to_pred = square_to_pred
        self.piece = piece

        self.score = self._calculate_score()

    def _calculate_score(self):
        from_square = chess.SQUARE_NAMES[self.move.from_square]
        to_square = chess.SQUARE_NAMES[self.move.to_square]

        if from_square in self.square_to_pred:
            return 0

        if to_square not in self.square_to_pred:
            return 0

        pred = self.square_to_pred[to_square]
        if pred.piece != self.piece:
            return 0

        return random.random()


class State:
    def __init__(self, keypoints):
        self.keypoints = keypoints

        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.node = self.game
        self.change = False

    def __repr__(self):
        return str(self.board)

    def get_image(self):
        bytestring = chess.svg.board(self.board, size=300)
        write_to = io.BytesIO()
        cairosvg.svg2png(bytestring=bytestring,
                         write_to=write_to)
        image = Image.open(write_to)
        return image

    def update(self, pred):
        self.change = False

        # piece_map = self.board.piece_map()
        # gt_squares = set([chess.SQUARE_NAMES[i] for i in piece_map.keys()])
        # pred_squares = set([p.square for p in pred])
        #
        # missing_detections = list(gt_squares - pred_squares)
        # new_detections = list(pred_squares - gt_squares)
        # ic(missing_detections)
        # ic(new_detections)
        square_to_pred = {p.square: p for p in pred}

        legal_moves = list(self.board.legal_moves)
        actions = [Action(move, square_to_pred,
                          PIECE_TO_CLASS[self.board.piece_at(move.from_square)])
                   for move in legal_moves]
        best_action = max(actions, key=lambda x: x.score)
        if best_action.score == 0:
            return

        ic(best_action.score)
        self.board.push(best_action.move)
        self.node = self.node.add_variation(best_action.move)
        self.change = True
