import json
import os
import shutil
from glob import glob

import numpy as np
from PIL import Image

from camera_chess.classifier import Classifier
from camera_chess.constants import CORNERS


def main():
    width, height = 900, 500
    keypoints = np.array([[177, 476], [16, 171], [574, 73], [884, 318]], dtype=np.float32)
    keypoints_template = {'original_width': width,
                          'original_height': height,
                          'from_name': 'kp-1',
                          'to_name': 'img-1',
                          'type': 'keypointlabels'}
    keypoints_labels = []
    for (x, y), square in zip(keypoints, CORNERS):
        keypoints_label = keypoints_template.copy()
        keypoints_label['value'] = {'x': 100 * x / width,
                                    'y': 100 * y / height,
                                    'width': 1.0,
                                    'keypointlabels': [square]}
        keypoints_labels.append(keypoints_label)

    classifier = Classifier(model_path='data/best.onnx',
                            conf_thres=0.1,
                            keypoints=keypoints)
    for image_path in glob('data/hikaru_sarin/images/*.jpg'):
        basename = os.path.basename(image_path)
        new_image_path = os.path.join('label_studio', 'files', 'images', basename)
        shutil.copyfile(image_path, new_image_path)

        d = {'data': {'img': f'/data/local-files/?d=images/{basename}'},
             'annotations': [{
                 'result': keypoints_labels.copy()
             }]}

        image = Image.open(image_path)
        pred = classifier.run(image)
        labels_template = {'original_width': width,
                           'original_height': height,
                           'from_name': 'bbox-1',
                           'to_name': 'img-1',
                           'type': 'rectanglelabels'}
        for p in pred:
            x = 100 * p.bbox[0] / width
            y = 100 * p.bbox[1] / height
            w = 100 * (p.bbox[2] - p.bbox[0]) / width
            h = 100 * (p.bbox[3] - p.bbox[1]) / height
            label = labels_template.copy()
            label['value'] = {'x': x, 'y': y, 'width': w, 'height': h, 'rectanglelabels': [p.piece]}
            d['annotations'][0]['result'].append(label)

        with open(os.path.join('label_studio', 'files', 'yolo', basename.replace('.jpg', '.json')), 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == '__main__':
    main()
