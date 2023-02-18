import io
import math

import cairosvg
import chess
import chess.pgn
import chess.svg
from PIL import Image

from camera_chess.constants import PIECE_TO_CLASS


class Action:
    def __init__(self, move, board):
        self.move = move
        self.board = board

        self.prev_center = None
        self.score = 0
        self.from_square = chess.SQUARE_NAMES[self.move.from_square]
        self.to_square = chess.SQUARE_NAMES[self.move.to_square]
        self.piece = PIECE_TO_CLASS[self.board.piece_at(self.move.from_square)]

    def update(self, square_to_pred):
        if self.from_square in square_to_pred:
            self.score = 0
            return

        if self.to_square not in square_to_pred:
            self.score = 0
            return

        pred = square_to_pred[self.to_square]
        if pred.piece != self.piece:
            self.score = 0
            return

        if self.prev_center is None:
            self.prev_center = pred.center
            self.score = 0
            return

        velocity = math.dist(self.prev_center, pred.center)
        self.score = 1


class State:
    def __init__(self, keypoints):
        self.keypoints = keypoints

        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.node = self.game
        self.change = False
        self.actions = [Action(move, self.board)
                        for move in list(self.board.legal_moves)]

    def __repr__(self):
        return str(self.board)

    def _play_move(self, move):
        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.actions = [Action(move, self.board)
                        for move in list(self.board.legal_moves)]
        self.change = True

    def get_image(self):
        bytestring = chess.svg.board(self.board, size=300)
        write_to = io.BytesIO()
        cairosvg.svg2png(bytestring=bytestring,
                         write_to=write_to)
        image = Image.open(write_to)
        return image

    def update(self, pred):
        self.change = False

        square_to_pred = {p.square: p for p in pred}
        for action in self.actions:
            action.update(square_to_pred)

        best_action = max(self.actions, key=lambda x: x.score)
        if best_action.score == 0:
            return

        self._play_move(best_action.move)


