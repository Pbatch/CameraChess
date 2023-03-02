import io

import cairosvg
import chess
import chess.pgn
import chess.svg
import numpy as np
from PIL import Image

from camera_chess.constants import PIECE_TO_CLASS, CLASSES


class State:
    ALPHA = 0.9

    def __init__(self, fen):
        self.fen = fen

        self.board = chess.Board(self.fen)
        self.game = chess.pgn.Game()
        self.node = self.game
        self.change = False
        self.last_move = None
        self.move_to_score = {}

        self.confs = self._set_confs()

    def _set_confs(self):
        confs = {}
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)

            p = np.zeros(len(CLASSES), dtype=np.float32)
            if piece is not None:
                p[CLASSES.index(PIECE_TO_CLASS[piece])] = 1.0
            confs[chess.square_name(square)] = p
        return confs

    def _play_move(self, move):
        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.change = True
        self.last_move = str(move)
        self._set_confs()

    def get_image(self):
        bytestring = chess.svg.board(self.board, size=300)
        write_to = io.BytesIO()
        cairosvg.svg2png(bytestring=bytestring,
                         write_to=write_to)
        image = Image.open(write_to)
        return image

    def update(self, pred):
        self.change = False

        missed = {chess.square_name(square) for square in chess.SQUARES}
        for p in pred:
            self.confs[p.square] = self.ALPHA * self.confs[p.square] + (1 - self.ALPHA) * p.confs
            missed.remove(p.square)
        for square in missed:
            self.confs[square] = self.ALPHA * self.confs[square]

        self.move_to_score = {}
        for move in list(self.board.legal_moves):
            from_square = str(move)[:2]
            to_square = str(move)[2:]
            piece = PIECE_TO_CLASS[self.board.piece_at(move.from_square)]
            piece_idx = CLASSES.index(piece)

            score = (1 - self.confs[from_square][piece_idx]) * self.confs[to_square][piece_idx]
            self.move_to_score[str(move)] = score
        self.move_to_score = {k: v for k, v in sorted(self.move_to_score.items(), key=lambda x: -x[1])}

        best_move, best_score = list(self.move_to_score.items())[0]
        if best_score > 0.05:
            self._play_move(chess.Move.from_uci(best_move))
