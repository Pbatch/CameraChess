import io

import cairosvg
import chess
import chess.pgn
import chess.svg
import numpy as np
from PIL import Image

from camera_chess.constants import PIECE_TO_CLASS, CLASSES


class Move:
    def __init__(self, move, speed_thresh=1.0, move_thresh=0.0):
        self.move = move
        self.speed_thresh = speed_thresh
        self.move_thresh = move_thresh

        self.from_square = str(move)[:2]
        self.to_square = str(move)[2:4]

        self.score = -1.0
        self.from_score = -1.0
        self.arrival_score = -1.0
        self.arrival_speed = 0.0

    def __repr__(self):
        d = {'from_score': self.from_score,
             'arrival_score': self.arrival_score,
             'arrival_speed': self.arrival_speed}
        return f'move={str(self.move)}, ' + ', '.join([f'{k}={v:.2f}' for k, v in d.items()])

    def uci(self):
        return self.move.uci()

    def set_score(self, board, square_to_scores, square_to_speed):
        to_piece = board.piece_at(self.move.from_square)
        if self.move.promotion is not None:
            to_piece = chess.Piece(piece_type=self.move.promotion,
                                   color=to_piece.color)

        self.from_score = np.max(square_to_scores[self.from_square])
        self.arrival_score = square_to_scores[self.to_square][CLASSES.index(PIECE_TO_CLASS[to_piece])]
        self.arrival_speed = square_to_speed[self.to_square]

        if self.from_score > 0:
            self.score = 0
        elif self.arrival_score <= self.move_thresh:
            self.score = 0
        elif self.arrival_speed > self.speed_thresh and not chess.Board.is_castling(board, self.move):
            self.score = 0
        else:
            self.score = self.arrival_score


class State:
    def __init__(self,
                 fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.fen = fen

        self.board = chess.Board(self.fen)
        self.game = chess.pgn.Game()
        self.node = self.game
        self.change = False
        self.last_move = None
        self.last_colour = 'black'

    def _play_move(self, move):
        self.board.push(move.move)
        self.node = self.node.add_variation(move.move)
        self.change = True
        self.last_move = move.uci()
        self.last_colour = ['white', 'black'][int(self.board.turn)]

    def get_image(self):
        bytestring = chess.svg.board(self.board, size=300)
        write_to = io.BytesIO()
        cairosvg.svg2png(bytestring=bytestring,
                         write_to=write_to)
        image = Image.open(write_to)
        return image

    def update(self, tracks):
        self.change = False

        square_to_scores = {chess.square_name(square): np.zeros(len(CLASSES), dtype=np.float32)
                            for square in chess.SQUARES}
        square_to_speed = {chess.square_name(square): 0.0
                           for square in chess.SQUARES}
        for track in tracks:
            piece_idx = CLASSES.index(track.piece)
            square_to_scores[track.square][piece_idx] = max(square_to_scores[track.square][piece_idx], track.score)
            square_to_speed[track.square] = max(square_to_speed[track.square], track.speed)

        moves = []
        for move_ in self.board.legal_moves:
            move = Move(move_)
            move.set_score(self.board, square_to_scores, square_to_speed)
            moves.append(move)

        valid_moves = [move for move in moves if move.score > 0]
        if len(valid_moves) == 0:
            return

        best_move = max(valid_moves, key=lambda x: x.score)

        clashing_moves = [move for move in valid_moves if move.to_square == best_move.to_square and move != best_move]
        if len(clashing_moves):
            return

        self._play_move(best_move)
