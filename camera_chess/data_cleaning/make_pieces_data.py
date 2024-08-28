import argparse
import json
import os
import subprocess
from glob import glob

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR, PIECES_DIR, DATASETS, EMPTY_DATASETS
from camera_chess.utils import clear_dir, get_roi, draw_text
from PIL import ImageDraw
from numba import njit


@njit(cache=True, fastmath=True)
def calculate_iomin(roi, bboxes):
    left_intersect = np.maximum(roi[0], bboxes[:, 0])
    top_intersect = np.maximum(roi[1], bboxes[:, 1])
    right_intersect = np.minimum(roi[2], bboxes[:, 2])
    bottom_intersect = np.minimum(roi[3], bboxes[:, 3])

    height_intersect = np.maximum(bottom_intersect - top_intersect, np.array(0.))
    width_intersect = np.maximum(right_intersect - left_intersect, np.array(0.))
    area_intersect = height_intersect * width_intersect

    area_bbox = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    iomin = area_intersect / area_bbox

    return iomin


def process_dataset(split, dataset, model_width, model_height, debug, single_class, need_keypoints,
                    iomin_threshold=0.5):
    train_size = max(model_width, model_height)

    label_paths = list(glob(os.path.join(DATA_DIR, dataset, 'labels', '*')))
    label_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    for label_path in tqdm(label_paths, desc=dataset):
        with open(label_path, 'r') as f:
            label = json.load(f)

        image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
        image = Image.open(image_path)
        id_ = f'{dataset.replace(os.path.sep, "_")}_{os.path.splitext(os.path.basename(image_path))[0]}'
        new_image_path = os.path.join(PIECES_DIR, split, 'images', f'{id_}.jpg')

        if dataset in EMPTY_DATASETS:
            image = ImageOps.contain(image, (train_size, train_size))
            image.save(new_image_path)
            continue

        if 'keypoints' in label:
            keypoints = np.array(list(label['keypoints'].values()))
        elif need_keypoints:
            continue
        else:
            keypoints = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        pieces = [bbox[0] for bbox in label['bboxes']]
        bboxes = np.array([bbox[1:] for bbox in label['bboxes']])

        try:
            keypoints[:, 0] *= image.width
            keypoints[:, 1] *= image.height
        except IndexError:
            print(f'Bad keypoints at "{label_path}". Skipping...')
            print(keypoints)
            continue
        roi = get_roi(keypoints, image.width, image.height, model_width, model_height)

        if len(bboxes):
            bboxes[:, [0, 2]] *= image.width
            bboxes[:, [1, 3]] *= image.height
            bboxes[:, [2, 3]] += bboxes[:, [0, 1]]
            iomin = calculate_iomin(roi, bboxes)
            bboxes = bboxes[iomin > iomin_threshold]

        image = image.crop(tuple(roi))
        crop_width, crop_height = image.width, image.height

        if len(bboxes):
            bboxes[:, [0, 2]] -= roi[0]
            bboxes[:, [1, 3]] -= roi[1]
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], a_min=0, a_max=crop_width)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], a_min=0, a_max=crop_height)

        image = ImageOps.contain(image, (train_size, train_size))
        image.save(new_image_path)

        output = []
        for piece, bbox in zip(pieces, bboxes):
            if single_class:
                class_id = 0
            else:
                class_id = CLASSES.index(piece)
            xc = (bbox[0] + bbox[2]) / (2 * crop_width)
            yc = (bbox[1] + bbox[3]) / (2 * crop_height)
            w = (bbox[2] - bbox[0]) / crop_width
            h = (bbox[3] - bbox[1]) / crop_height
            output.append(f'{class_id} {xc} {yc} {w} {h}')
        new_label_path = os.path.join(PIECES_DIR, split, 'labels', f'{id_}.txt')
        with open(new_label_path, 'w') as f:
            f.write('\n'.join(output))

        if debug:
            d = ImageDraw.Draw(image)
            for s in output:
                xc, yc, w, h = [float(i) for i in s.split()[1:]]
                bbox = [(xc - w / 2) * image.width,
                        (yc - h / 2) * image.height,
                        (xc + w / 2) * image.width,
                        (yc + h / 2) * image.height]
                draw_text(d, bbox, 'label')

            image.show()
            input()


def main(model_width, model_height, debug, single_class):
    for split in DATASETS.keys():
        for i in ['images', 'labels']:
            clear_dir(os.path.join(PIECES_DIR, split, i))

    for split, datasets in DATASETS.items():
        need_keypoints = split != 'synthetic'
        for dataset in datasets:
            process_dataset(split, dataset, model_width, model_height, debug, single_class, need_keypoints)

    tar_dir = os.path.relpath(PIECES_DIR, DATA_DIR)
    subprocess.call(['tar', '-czf', f'{tar_dir}.tar.gz', tar_dir], cwd=DATA_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mw', '--model_width', type=int, default=480)
    parser.add_argument('-mh', '--model_height', type=int, default=288)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-s', '--single_class', action='store_true')
    args = parser.parse_args()
    main(args.model_width, args.model_height, args.debug, args.single_class)
