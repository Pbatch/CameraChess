import os
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LABEL_STUDIO_DIR = os.path.join(ROOT_DIR, 'label_studio')
UPLOAD_DIR = os.path.join(LABEL_STUDIO_DIR, 'data', 'media', 'upload')
CLASSES = ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
           'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook',
           'empty']
CLASS_TO_PIECE = {'black-bishop': chess.Piece(chess.BISHOP, chess.BLACK),
                  'black-king': chess.Piece(chess.KING, chess.BLACK),
                  'black-knight': chess.Piece(chess.KNIGHT, chess.BLACK),
                  'black-pawn': chess.Piece(chess.PAWN, chess.BLACK),
                  'black-queen': chess.Piece(chess.QUEEN, chess.BLACK),
                  'black-rook': chess.Piece(chess.ROOK, chess.BLACK),
                  'white-bishop': chess.Piece(chess.BISHOP, chess.WHITE),
                  'white-king': chess.Piece(chess.KING, chess.WHITE),
                  'white-knight': chess.Piece(chess.KNIGHT, chess.WHITE),
                  'white-pawn': chess.Piece(chess.PAWN, chess.WHITE),
                  'white-queen': chess.Piece(chess.QUEEN, chess.WHITE),
                  'white-rook': chess.Piece(chess.ROOK, chess.WHITE)}
PIECE_TO_CLASS = {v: k for k, v in CLASS_TO_PIECE.items()}
CORNERS = ['h1', 'a1', 'a8', 'h8']
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE
COLOUR_MAP = {'black-pawn': 'white',
              'white-pawn': 'white',
              'black-knight': 'blue',
              'white-knight': 'blue',
              'black-bishop': 'red',
              'white-bishop': 'red',
              'black-rook': 'yellow',
              'white-rook': 'yellow',
              'black-king': 'black',
              'white-king': 'black',
              'black-queen': 'grey',
              'white-queen': 'grey'}
