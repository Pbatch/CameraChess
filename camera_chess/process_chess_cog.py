import json
import os
from glob import glob

from PIL import Image
from tqdm import tqdm

from camera_chess.constants import ROOT_DIR


def main():
    names = ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop',
             'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']
    cog_map = {'k': 'black-king',
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
    roboflow_dir = os.path.join(ROOT_DIR, 'data', 'chesscog', 'roboflow')
    for old_label_path in tqdm(sorted(glob('data/chesscog/*/*/*.json'))):
        old_image_path = old_label_path.replace('.json', '.png')
        basename = os.path.basename(old_image_path)

        new_image_path = os.path.join(roboflow_dir, 'images', basename.replace('.png', '.jpg'))
        image = Image.open(old_image_path).convert('RGB')
        image.save(new_image_path)

        with open(old_label_path) as f:
            d = json.load(f)

        new_label_path = os.path.join(roboflow_dir, 'labels', basename.replace('.png', '.txt'))
        with open(new_label_path, 'w') as f:
            for piece in d['pieces']:
                label = cog_map[piece['piece']]
                bbox = piece['box']
                x = bbox[0] / image.width
                y = bbox[1] / image.height
                w = bbox[2] / image.width
                h = bbox[3] / image.height
                x += w / 2
                y += h / 2
                f.write(f'{names.index(label)} {x} {y} {w} {h}\n')


if __name__ == '__main__':
    main()
