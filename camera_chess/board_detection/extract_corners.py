import argparse
import json
import os
import subprocess
from glob import glob

import numpy as np
from PIL import ImageOps, Image
from tqdm import tqdm

from camera_chess.constants import SQUARE_SIZE, YOLO_DIR, DATA_DIR
from camera_chess.data_cleaning.make_yolo_data import DATASETS
from camera_chess.utils import clear_dir, warp


def process_dataset(split, dataset, image_size):
    label_paths = list(glob(os.path.join(DATA_DIR, dataset, 'labels', '*')))
    for label_path in tqdm(label_paths, desc=dataset):
        with open(label_path, 'r') as f:
            label = json.load(f)
        try:
            keypoints = np.array(list(label['keypoints'].values()), dtype=np.float32)
        except KeyError:
            print(f'No keypoints for label path {label_path}')
            continue

        image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
        id_ = f'{dataset.replace(os.path.sep, "_")}_{os.path.splitext(os.path.basename(image_path))[0]}'
        new_image_path = os.path.join(YOLO_DIR, split, 'images', f'{id_}.jpg')

        image = Image.open(image_path)
        image = ImageOps.contain(image, (image_size, image_size))
        image.save(new_image_path)

        grid = np.mgrid[1:8, 1:8].reshape(2, -1).T.astype(np.float32) * SQUARE_SIZE
        square_centers = warp(grid, keypoints)
        max_corner_dist = np.max(np.sqrt(np.sum(np.square(np.diff([*keypoints, keypoints[0]], axis=0)), axis=1)))

        class_id = 0
        w = max_corner_dist / 24
        h = max_corner_dist / 24
        ratio = image.width / image.height
        if ratio > 1:
            w /= ratio
        else:
            h /= ratio
        output = [f"{class_id} {xc} {yc} {w} {h}" for xc, yc in square_centers]
        new_label_path = os.path.join(YOLO_DIR, split, 'labels', f'{id_}.txt')
        with open(new_label_path, 'w') as f:
            f.write('\n'.join(output))


def main(image_size):
    for split in DATASETS.keys():
        for i in ['images', 'labels']:
            clear_dir(os.path.join(YOLO_DIR, split, i))

    for split, datasets in DATASETS.items():
        for dataset in datasets:
            process_dataset(split, dataset, image_size)

    subprocess.call(['tar', '-czf', 'yolo.tar.gz', 'yolo'], cwd=DATA_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--image_size', type=int, default=480)
    args = parser.parse_args()
    main(args.image_size)
