import argparse
import json
import os
import shutil

from imagesize import imagesize
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR, STUDIO_IMAGE_DIR, STUDIO_LABEL_DIR
from glob import glob

from camera_chess.utils import clear_dir


def main(dataset):
    clear_dir(STUDIO_IMAGE_DIR)
    clear_dir(STUDIO_LABEL_DIR)

    image_paths = sorted(glob(os.path.join(DATA_DIR, *dataset.split('/'), 'images', '*.jpg')))

    for image_path in tqdm(image_paths):
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.json')

        basename = os.path.basename(image_path)
        new_image_path = os.path.join(STUDIO_IMAGE_DIR, basename)
        shutil.copyfile(image_path, new_image_path)
        width, height = imagesize.get(image_path)

        with open(label_path, 'rb') as f:
            labels = json.load(f)

        keypoints_labels = []
        if 'keypoints' in labels:
            keypoints_template = {'original_width': width,
                                  'original_height': height,
                                  'from_name': 'kp-1',
                                  'to_name': 'img-1',
                                  'type': 'keypointlabels'}
            for square, (x, y) in labels['keypoints'].items():
                keypoints_label = keypoints_template.copy()
                keypoints_label['value'] = {'x': 100 * x,
                                            'y': 100 * y,
                                            'width': 1.0,
                                            'keypointlabels': [square]}
                keypoints_labels.append(keypoints_label)

        d = {'data': {'img': f'/data/local-files/?d=images/{basename}'},
             'annotations': [{
                 'result': keypoints_labels
             }]}

        result = {'original_width': width,
                  'original_height': height,
                  'from_name': 'bbox-1',
                  'to_name': 'img-1',
                  'type': 'rectanglelabels'}

        for class_, x, y, w, h in labels['bboxes']:
            label = result.copy()
            label['value'] = {'x': 100 * x, 'y': 100 * y, 'width': 100 * w, 'height': 100 * h,
                              'rectanglelabels': [class_]}
            d['annotations'][0]['result'].append(label)

        with open(os.path.join(STUDIO_LABEL_DIR, basename.replace('.jpg', '.json')), 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
