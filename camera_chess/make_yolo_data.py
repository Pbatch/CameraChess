import argparse
import json
import os
import shutil
import subprocess
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import ROOT_DIR, CLASSES


def main(image_size, train_fraction):
    for split in ['train', 'val']:
        for i in ['images', 'labels']:
            p = os.path.join('data', 'yolo', split, i)
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p)

    for dataset in ['google', 'roboflow_1', 'roboflow_2', 'roboflow_3', 'roboflow_5']:
        for label_path in tqdm(glob(os.path.join('data', dataset, 'labels', '*'))):
            with open(label_path, 'r') as f:
                label = json.load(f)

            image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
            image = np.array(Image.open(image_path))
            image_height, image_width = image.shape[:2]

            keypoints = np.array(list(label['keypoints'].values()))
            pieces = [bbox[0] for bbox in label['bboxes']]
            bboxes = np.array([bbox[1:] for bbox in label['bboxes']])

            x_min = keypoints[:, 0].min()
            y_min = keypoints[:, 1].min()
            x_max = keypoints[:, 0].max()
            y_max = keypoints[:, 1].max()
            try:
                bboxes[:, [0, 2]] *= image_width
                bboxes[:, [1, 3]] *= image_height
                x_min = min(bboxes[:, 0].min(), x_min)
                y_min = min(bboxes[:, 1].min(), y_min)
                x_max = max((bboxes[:, 0] + bboxes[:, 2]).max(), x_max)
                y_max = max((bboxes[:, 1] + bboxes[:, 3]).max(), y_max)
                bboxes[:, 0] -= x_min
                bboxes[:, 1] -= y_min
            except IndexError:
                print(f'No bboxes for image {image_path}')

            crop = Image.fromarray(image).crop((x_min, y_min, x_max, y_max))
            crop_width, crop_height = crop.width, crop.height
            crop = crop.resize((image_size, image_size))
            split = "train" if np.random.random() < train_fraction else "val"
            id_ = f'{dataset}_{os.path.splitext(os.path.basename(image_path))[0]}'
            new_image_path = os.path.join('data', 'yolo', split, 'images', f'{id_}.jpg')
            crop.save(new_image_path)

            output = []
            for piece, bbox in zip(pieces, bboxes):
                class_id = CLASSES.index(piece)
                xc = (bbox[0] + bbox[2] / 2) / crop_width
                yc = (bbox[1] + bbox[3] / 2) / crop_height
                w = bbox[2] / crop_width
                h = bbox[3] / crop_height
                output.append(f'{class_id} {xc} {yc} {w} {h}')
            new_label_path = os.path.join('data', 'yolo', split, 'labels', f'{id_}.txt')
            with open(new_label_path, 'w') as f:
                f.write('\n'.join(output))
    subprocess.call(['tar', '-czf', 'yolo.tar.gz', 'yolo'], cwd=os.path.join(ROOT_DIR, 'data'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--image_size', type=int, default=640)
    parser.add_argument('-f', '--train_fraction', type=float, default=0.9)
    args = parser.parse_args()
    main(args.image_size, args.train_fraction)
