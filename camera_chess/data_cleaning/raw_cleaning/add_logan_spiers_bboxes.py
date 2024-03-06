import argparse
import json
import os
from glob import glob

import numpy as np

from camera_chess.constants import DATA_DIR, CORNERS, CLASSES
from camera_chess.detector import Detector
from PIL import Image, ImageDraw

from camera_chess.sequence_generator import get_centers_and_boundary, process_preds
from camera_chess.utils import draw_lines, draw_text


def main(start, end, debug):
    split = f'{start}-{end - 1}'
    model_basename = "480S_pieces_480x288.onnx"
    detector = Detector(model_basename=model_basename)

    for image_path in glob(os.path.join(DATA_DIR, 'logan_spiers', split, 'images', '*.jpg')):
        image = np.array(Image.open(image_path))

        label_path = os.path.join(DATA_DIR, 'logan_spiers', split,
                                  'labels', os.path.basename(image_path).replace('.jpg', '.json'))
        with open(label_path) as f:
            label = json.load(f)
        keypoints = label['keypoints']
        for key in keypoints.keys():
            keypoints[key][0] *= image.shape[1]
            keypoints[key][1] *= image.shape[0]
            keypoints[key] = np.array(keypoints[key])
        preds = detector.run(image, keypoints)

        conf = np.max(preds[:, 4:], axis=1)
        conf_mask = conf > 0.1
        preds = preds[conf_mask]
        conf = conf[conf_mask]

        centers, boundary = get_centers_and_boundary([keypoints[k] for k in CORNERS])
        frame_info, frame_boxes = process_preds(preds, conf, boundary, centers)

        if debug:
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image)
            draw_lines(d, list(keypoints.values()), colour='red')
            for text, (x, y) in keypoints.items():
                bbox = [x - 5, y - 5, x + 5, y + 5]
                draw_text(d, bbox, text)

            for info in frame_info:
                bbox = info[2:6]
                cls = int(info[6])
                conf = info[7]
                text = f'{CLASSES[cls]}:{conf:.2f}'
                draw_text(d, bbox, text=text)
            pil_image.show()
            input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int, required=True)
    parser.add_argument('--end', '-e', type=int, required=True)
    parser.add_argument('--debug', '-d', action='store_true')
    args = parser.parse_args()
    main(args.start, args.end, args.debug)
