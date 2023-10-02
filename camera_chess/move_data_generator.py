import chess

from camera_chess.constants import CLASSES, PIECE_TO_CLASS


class MoveDataGenerator:
    CASTLING_MAP = {chess.parse_square("g1"):
                        [chess.parse_square("h1"), chess.parse_square("f1"), CLASSES.index("white-rook")],
                    chess.parse_square("c1"):
                        [chess.parse_square("a1"), chess.parse_square("d1"), CLASSES.index("white-rook")],
                    chess.parse_square("g8"):
                        [chess.parse_square("h8"), chess.parse_square("f8"), CLASSES.index("black-rook")],
                    chess.parse_square("c8"):
                        [chess.parse_square("a8"), chess.parse_square("d8"), CLASSES.index("black-rook")]}

    @staticmethod
    def _get_piece_idx(move, board):
        if move.promotion is None:
            piece = board.piece_at(move.from_square)
        else:
            piece = move.promotion
        piece_idx = CLASSES.index(PIECE_TO_CLASS[piece])
        return piece_idx

    def _get_data(self, move, board):
        from_squares = [move.from_square]
        to_squares = [move.to_square]
        targets = [self._get_piece_idx(move, board)]
        if board.is_castling(move):
            from_square, to_square, target = self.CASTLING_MAP[move.to_square]
            from_squares.append(from_square)
            to_squares.append(to_square)
            targets.append(target)
        elif board.is_en_passant(move):
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            captured_pawn_square = chess.parse_square(to_square[0] + from_square[1])
            from_squares.append(captured_pawn_square)
        d = {
            'san': board.san(move),
            'from': from_squares,
            'to': to_squares,
            'targets': targets
        }
        return d

    def _combine_data(self, move1_data, move2_data):
        bad_squares = move2_data['from'] + move2_data['to']

        from1 = list(set(move1_data['from']) - set(bad_squares))

        to1 = [target for i, target in enumerate(move1_data['to']) if move1_data['to'][i] not in bad_squares]
        targets1 = [target for i, target in enumerate(move1_data['targets']) if move1_data['to'][i] not in bad_squares]

        from_squares = from1 + move2_data['from']
        to_squares = to1 + move2_data['to']
        targets = targets1 + move2_data['targets']
        d = {
            'san': [move1_data['san'], move2_data['san']],
            'from': from_squares,
            'to': to_squares,
            'targets': targets
        }
        return d

    def run(self, board):
        move_data = []
        for move1 in board.legal_moves:
            move1_data = self._get_data(move1, board)
            board.push(move1)
            done = True
            for move2 in board.legal_moves:
                move2_data = self._get_data(move2, board)
                moves_data = self._combine_data(move1_data, move2_data)
                d = {
                    "move1": move1_data,
                    "move2": move2_data,
                    "moves": moves_data
                }
                move_data.append(d)
                done = False
            if done:
                d = {
                    "move1": move1_data
                }
                move_data.append(d)
            board.pop()
        return move_data
