import argparse
import json
import os
import random
import subprocess
from glob import glob

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR, YOLO_DIR
from camera_chess.utils import clear_dir


def main(image_size, train_fraction, max_split_size):
    for split in ['train', 'val']:
        for i in ['images', 'labels']:
            clear_dir(os.path.join(YOLO_DIR, split, i))

    youtube_datasets = ['hikaru_sarin', 'dubov_nepo', 'carlsen_toma', 'carlsen_vidit', 'carlsen_abdu',
                        'shimanov_vidit', 'harika_nana', 'harika_mariam', 'anand_carlsen', 'gukesh_shakh',
                        'karayaman', 'hikaru_vasif', 'magnus_madaminov', 'hari_tuan', 'hans_rinat']
    roboflow_datasets = ['1', '2', '3', '4', '5']
    peter_datasets = ['smothered_mate', 'scholars_mate', 'kasparov_immortal',
                      'peter_emma', 'wells_shirov', 'gerasimov_smyslov', 'bronstein_teschner',
                      'melgosa_zuluaga', 'campora_morozevich', 'eingorn_vaganian',
                      'tal_sviridov', 'larsen_spassky', 'furman_spassky', 'wells_speelman']
    single_piece_datasets = ['white-rook', 'black-rook', 'white-queen', 'black-queen',
                             'white-pawn', 'black-pawn', 'white-king', 'black-king',
                             'white-bishop', 'black-bishop', 'white-knight', 'black-knight']
    datasets = ['google', 'chesscog']
    datasets.extend([os.path.join('peter', s) for s in peter_datasets])
    datasets.extend([os.path.join('youtube', s) for s in youtube_datasets])
    datasets.extend([os.path.join('roboflow', s) for s in roboflow_datasets])
    datasets.extend([os.path.join('single_piece', s) for s in single_piece_datasets])
    for dataset in datasets:
        label_paths = list(glob(os.path.join(DATA_DIR, dataset, 'labels', '*')))
        if len(label_paths) > max_split_size:
            label_paths = random.sample(label_paths, max_split_size)
        for label_path in tqdm(label_paths, desc=dataset):
            with open(label_path, 'r') as f:
                label = json.load(f)

            image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
            image = Image.open(image_path)

            try:
                keypoints = np.array(list(label['keypoints'].values()))
            except KeyError:
                keypoints = None
            pieces = [bbox[0] for bbox in label['bboxes']]
            bboxes = np.array([bbox[1:] for bbox in label['bboxes']])

            if keypoints is not None:
                keypoints[:, 0] *= image.width
                keypoints[:, 1] *= image.height
                x_min = keypoints[:, 0].min()
                y_min = keypoints[:, 1].min()
                x_max = keypoints[:, 0].max()
                y_max = keypoints[:, 1].max()
            else:
                x_min = 0
                y_min = 0
                x_max = image.width
                y_max = image.height

            if len(bboxes):
                bboxes[:, [0, 2]] *= image.width
                bboxes[:, [1, 3]] *= image.height
                x_min = max(min(bboxes[:, 0].min(), x_min), 0)
                y_min = max(min(bboxes[:, 1].min(), y_min), 0)
                x_max = min(max((bboxes[:, 0] + bboxes[:, 2]).max(), x_max), image.width)
                y_max = min(max((bboxes[:, 1] + bboxes[:, 3]).max(), y_max), image.height)
                bboxes[:, 0] -= x_min
                bboxes[:, 1] -= y_min

            image = image.crop((x_min, y_min, x_max, y_max))
            crop_width, crop_height = image.width, image.height
            image = ImageOps.contain(image, (image_size, image_size))

            split = "train" if np.random.random() < train_fraction else "val"
            id_ = f'{dataset.replace(os.path.sep, "_")}_{os.path.splitext(os.path.basename(image_path))[0]}'
            new_image_path = os.path.join(YOLO_DIR, split, 'images', f'{id_}.jpg')
            image.save(new_image_path)

            output = []
            for piece, bbox in zip(pieces, bboxes):
                class_id = CLASSES.index(piece)
                xc = (bbox[0] + bbox[2] / 2) / crop_width
                yc = (bbox[1] + bbox[3] / 2) / crop_height
                w = bbox[2] / crop_width
                h = bbox[3] / crop_height
                output.append(f'{class_id} {xc} {yc} {w} {h}')
            new_label_path = os.path.join(YOLO_DIR, split, 'labels', f'{id_}.txt')
            with open(new_label_path, 'w') as f:
                f.write('\n'.join(output))

    subprocess.call(['tar', '-czf', 'yolo.tar.gz', 'yolo'], cwd=DATA_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--image_size', type=int, default=480)
    parser.add_argument('-f', '--train_fraction', type=float, default=0.9)
    parser.add_argument('-m', '--max_split_size', type=int, default=2000)
    args = parser.parse_args()
    main(args.image_size, args.train_fraction, args.max_split_size)
