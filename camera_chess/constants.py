import os
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STUDIO_DIR = os.path.join(ROOT_DIR, 'label_studio')
STUDIO_IMAGE_DIR = os.path.join(STUDIO_DIR, 'files', 'images')
STUDIO_LABEL_DIR = os.path.join(STUDIO_DIR, 'files', 'labels')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
YOLO_DIR = os.path.join(DATA_DIR, 'yolo')
KEYPOINTS_DIR = os.path.join(DATA_DIR, 'keypoints')

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
CLASSES = list(CLASS_TO_PIECE.keys())
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
ABBR_MAP = {'black-pawn': 'p',
            'white-pawn': 'P',
            'black-knight': 'n',
            'white-knight': 'N',
            'black-bishop': 'b',
            'white-bishop': 'B',
            'black-rook': 'r',
            'white-rook': 'R',
            'black-king': 'k',
            'white-king': 'K',
            'black-queen': 'q',
            'white-queen': 'Q'}
SQUARE_TO_PIECE = {**{f'{i}2': 'white-pawn' for i in 'abcdefgh'},
                   **{f'{i}7': 'black-pawn' for i in 'abcdefgh'},
                   'a1': 'white-rook',
                   'h1': 'white-rook',
                   'a8': 'black-rook',
                   'h8': 'black-rook',
                   'b1': 'white-knight',
                   'g1': 'white-knight',
                   'b8': 'black-knight',
                   'g8': 'black-knight',
                   'c1': 'white-bishop',
                   'f1': 'white-bishop',
                   'c8': 'black-bishop',
                   'f8': 'black-bishop',
                   'd8': 'black-queen',
                   'e8': 'black-king',
                   'd1': 'white-queen',
                   'e1': 'white-king'}

"""
Lichess Board API
"""
# username = "cameraboard1"
LICHESS_TOKEN_1 = "lip_mYMY7izVDkuUnIssVYxE"

# username = "cameraboard2"
LICHESS_TOKEN_2 = "lip_w1lKuk3GbV7eHjo0KFXL"

# username = "tom24008"
LICHESS_TOKEN_TOM = "lip_qqlkc4ArNe0WuXM7JLxQ"

# username = "blindfoldblunderer"
LICHESS_TOKEN_PETER = "lip_3Km8pHjYCeNkTRL7HODE"

# username = "babyeatingbishop"
LICHESS_TOKEN_CONOR = "lip_KHK4q3qAH5TFUfbX83zP"
