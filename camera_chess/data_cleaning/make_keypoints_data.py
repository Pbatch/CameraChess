import argparse
import json
import os
import subprocess
from glob import glob

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from camera_chess.constants import DATA_DIR, KEYPOINTS_DIR
from camera_chess.data_cleaning.make_yolo_data import DATASETS
from camera_chess.utils import clear_dir


def process_dataset(split, dataset, image_size):
    label_paths = list(glob(os.path.join(DATA_DIR, dataset, 'labels', '*')))
    for label_path in tqdm(label_paths, desc=dataset):
        with open(label_path, 'r') as f:
            label = json.load(f)
        if 'keypoints' not in label:
            continue

        image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
        id_ = f'{dataset.replace(os.path.sep, "_")}_{os.path.splitext(os.path.basename(image_path))[0]}'
        new_image_path = os.path.join(KEYPOINTS_DIR, split, 'images', f'{id_}.jpg')
        image = Image.open(image_path)
        image = ImageOps.contain(image, (image_size, image_size))
        image.save(new_image_path)
        values = np.array(list(label['keypoints'].values())).clip(0, 1)
        x_min = values[:, 0].min()
        y_min = values[:, 1].min()
        x_max = values[:, 0].max()
        y_max = values[:, 1].max()

        class_index = 0
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        keypoints = [j for key in ['h1', 'a1', 'a8', 'h8'] for j in label['keypoints'][key]]
        keypoints = np.array(keypoints).clip(0, 1).tolist()

        # <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
        keypoint_string = ' '.join([str(i) for i in [class_index, x, y, width, height, *keypoints]])

        new_label_path = os.path.join(KEYPOINTS_DIR, split, 'labels', f'{id_}.txt')
        with open(new_label_path, 'w') as f:
            f.write(keypoint_string)


def main(image_size):
    for split in DATASETS.keys():
        for i in ['images', 'labels']:
            clear_dir(os.path.join(KEYPOINTS_DIR, split, i))

    for split, datasets in DATASETS.items():
        if split == 'synthetic':
            continue
        for dataset in datasets:
            process_dataset(split, dataset, image_size)

    subprocess.call(['tar', '-czf', 'keypoints.tar.gz', 'keypoints'], cwd=DATA_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--image_size', type=int, default=480)
    args = parser.parse_args()
    main(args.image_size)
