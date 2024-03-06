import json
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import CORNERS, DATA_DIR, CHAR_TO_CATEGORY


def main():
    for i, old_label_path in tqdm(enumerate(sorted(glob('data/chesscog/raw/*/*.json')))):
        old_image_path = old_label_path.replace('.json', '.png')

        new_image_path = os.path.join(DATA_DIR, 'chesscog', 'images', f'{i}.jpg')
        image = Image.open(old_image_path).convert('RGB')
        width, height = image.width, image.height
        image.save(new_image_path)

        new_label_path = os.path.join(DATA_DIR, 'chesscog', 'labels', f'{i}.json')
        with open(old_label_path) as f:
            d = json.load(f)
        keypoints = {s: [c[0] / width, c[1] / height]
                     for s, c in zip(CORNERS, np.roll(d['corners'], shift=3, axis=0))}

        bboxes = []
        for piece in d['pieces']:
            label = CHAR_TO_CATEGORY[piece['piece']]
            bbox = piece['box']
            x = bbox[0] / image.width
            y = bbox[1] / image.height
            w = bbox[2] / image.width
            h = bbox[3] / image.height
            bboxes.append([label, x, y, w, h])

        label = {'keypoints': keypoints,
                 'bboxes': bboxes}
        with open(new_label_path, 'w') as f:
            json.dump(label, f, indent=4)


if __name__ == '__main__':
    main()
