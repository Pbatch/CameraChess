import os
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STUDIO_DIR = os.path.join(ROOT_DIR, 'label_studio')
STUDIO_IMAGE_DIR = os.path.join(STUDIO_DIR, 'files', 'images')
STUDIO_LABEL_DIR = os.path.join(STUDIO_DIR, 'files', 'labels')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
PIECES_DIR = os.path.join(DATA_DIR, 'pieces')
XCORNERS_DIR = os.path.join(DATA_DIR, 'xcorners')

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
ABBR_MAP = {'black-pawn': 'BP',
            'white-pawn': 'WP',
            'black-knight': 'BN',
            'white-knight': 'WN',
            'black-bishop': 'BB',
            'white-bishop': 'WB',
            'black-rook': 'BR',
            'white-rook': 'WR',
            'black-king': 'BK',
            'white-king': 'WK',
            'black-queen': 'BQ',
            'white-queen': 'WQ'}
CHAR_TO_CATEGORY = {'k': 'black-king',
                    'q': 'black-queen',
                    'r': 'black-rook',
                    'p': 'black-pawn',
                    'b': 'black-bishop',
                    'n': 'black-knight',
                    'K': 'white-king',
                    'Q': 'white-queen',
                    'R': 'white-rook',
                    'P': 'white-pawn',
                    'B': 'white-bishop',
                    'N': 'white-knight'}
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

# Could add chessvision/train to synthetic data (200k images...)
# TODO: How do we add back empty images like "Google Empty" with no keypoints data? (single piece are similar)
# TODO: Label corners for all datasets
DATASETS = {
    # 'synthetic': ['chesscog',
    #               'chessred2k',
    #               *[os.path.join('chessvision', s) for s in ['test']],
    #               *[os.path.join('roboflow', s) for s in ['7', '10']]
    #               ],
    'val': ['google',
            os.path.join('peter', 'scholars_mate'),
            os.path.join('youtube', 'carlsen_vidit'),
            os.path.join('four_corners', 'caro'),
            os.path.join('mercato', 'english')],
    'train': [*[os.path.join('youtube', s) for s in ['hikaru_sarin', 'dubov_nepo', 'carlsen_toma', 'shimanov_vidit',
                                                     'harika_nana', 'anand_carlsen', 'gukesh_shakh',
                                                     'karayaman', 'hikaru_vasif', 'magnus_madaminov', 'hari_tuan',
                                                     'hans_rinat', 'retired_lawyer', 'ramirez_yoo']],
              *[os.path.join('peter', s) for s in ['smothered_mate', 'kasparov_immortal', 'peter_emma',
                                                   'wells_shirov', 'gerasimov_smyslov', 'bronstein_teschner',
                                                   'melgosa_zuluaga', 'campora_morozevich', 'eingorn_vaganian',
                                                   'tal_sviridov', 'larsen_spassky', 'furman_spassky',
                                                   'wells_speelman', 'glass_1', 'glass_2', 'glass_3', 'pub']],
              *[os.path.join('big_stand', s) for s in ['slav', 'berlin']],
              *[os.path.join('small_stand', s) for s in ['slav', 'berlin']],
              *[os.path.join('four_corners', s) for s in ['french', 'london', 'ponziani', 'tromp']],
              *[os.path.join('mercato', s) for s in ['bogdan', 'elephant', 'james', 'reti', 'gambit',
                                                     'ruy', 'slav', 'vienna']],
              *[os.path.join('roboflow', s) for s in ['1', '2', '3', '4', '5', '6', '8', '9', '11',
                                                      'public', 'ppp']],
              *[os.path.join('peter_wooden', s) for s in ['scholars_mate', 'smothered_mate', 'gerasimov_smyslov',
                                                          'wells_shirov']],
              *[os.path.join('tom', s) for s in ['slav', 'italian', 'spanish']]
              ]
}