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
