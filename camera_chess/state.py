import io

import cairosvg
import chess
import chess.pgn
import chess.svg
from PIL import Image

from camera_chess import constants


class Action:
    def __init__(self, move, board, min_hits):
        self.move = move
        self.board = board
        self.min_hits = min_hits

        self.hits = 0
        self.score = 0
        self.error = ''
        self.from_square = chess.SQUARE_NAMES[self.move.from_square]
        self.to_square = chess.SQUARE_NAMES[self.move.to_square]
        self.piece = constants.PIECE_TO_CLASS[self.board.piece_at(self.move.from_square)]

    def __repr__(self):
        return f'{self.move} {self.score} {self.error}'

    def update(self, square_to_pred, square_to_gt):
        self.score = 0
        self.error = ''
        if self.from_square in square_to_pred:
            self.error = f'Piece did not leave {self.from_square}'
            return

        if self.to_square not in square_to_pred:
            self.error = f'Piece did not arrive at {self.to_square}'
            return

        if self.to_square in square_to_gt:
            pred = square_to_pred[self.to_square]
            if pred.piece != self.piece:
                self.error = f'Wrong piece classification at {self.to_square} ({pred.piece} != {self.piece})'
                return

        self.hits += 1
        if self.hits < self.min_hits:
            self.error = 'Not enough hits'
            return

        if self.piece in {'white-king', 'black-king'}:
            self.score = 2
        else:
            self.score = 1


class State:
    def __init__(self, keypoints, fen, min_hits=2):
        self.keypoints = keypoints
        self.fen = fen
        self.min_hits = min_hits

        self.board = chess.Board(fen=self.fen)
        self.game = chess.pgn.Game()
        self.node = self.game
        self.change = False
        self.last_move = None
        self.actions = []

        self._reset_actions()

    def __repr__(self):
        return str(self.board)

    def _reset_actions(self):
        self.actions = [Action(move, self.board, self.min_hits)
                        for move in list(self.board.legal_moves)]

    def _play_move(self, move):
        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.change = True
        self.last_move = str(move)
        self._reset_actions()

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
        square_to_gt = {}
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                square_to_gt[chess.square_name(square)] = constants.PIECE_TO_CLASS[piece]

        while True:
            for action in self.actions:
                action.update(square_to_pred, square_to_gt)

            best_action = max(self.actions, key=lambda x: x.score)
            if best_action.score == 0:
                break

            self._play_move(best_action.move)

    def debug(self, move, pred):
        square_to_piece = {p.square: p.piece for p in pred}

        from_square = move[:2]
        from_piece = square_to_piece.get(from_square, "missing")

        to_square = move[2:]
        to_piece = square_to_piece.get(to_square, "missing")

        print(f'({from_square}, {to_square}): ({from_piece}, {to_piece})')
