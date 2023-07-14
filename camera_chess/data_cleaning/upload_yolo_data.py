import json
import os
import shutil

from imagesize import imagesize
from tqdm import tqdm

from camera_chess.constants import CLASSES, DATA_DIR, STUDIO_IMAGE_DIR, STUDIO_LABEL_DIR
from glob import glob

from camera_chess.utils import clear_dir


def main():
    dataset_id = 11
    image_paths = sorted(glob(os.path.join(DATA_DIR, 'roboflow', str(dataset_id), 'images', '*.jpg')))
    clear_dir(STUDIO_IMAGE_DIR)
    clear_dir(STUDIO_LABEL_DIR)

    for image_path in tqdm(image_paths):
        yolo_path = image_path.replace('images', 'yolov5').replace('.jpg', '.txt')

        basename = os.path.basename(image_path)
        new_image_path = os.path.join(STUDIO_IMAGE_DIR, basename)
        shutil.copyfile(image_path, new_image_path)

        d = {'data': {'img': f'/data/local-files/?d=images/{basename}'},
             'annotations': [{
                 'result': []
             }]}

        width, height = imagesize.get(image_path)
        result = {'original_width': width,
                  'original_height': height,
                  'from_name': 'bbox-1',
                  'to_name': 'img-1',
                  'type': 'rectanglelabels'}
        with open(yolo_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]

        try:
            for class_id, xc, yc, w, h in lines:
                class_ = CLASSES[int(class_id)]
                xc, yc, w, h = [float(i) for i in [xc, yc, w, h]]
                x = 100 * (xc - w/2)
                y = 100 * (yc - h/2)
                w = 100 * w
                h = 100 * h
                label = result.copy()
                label['value'] = {'x': x, 'y': y, 'width': w, 'height': h, 'rectanglelabels': [class_]}
                d['annotations'][0]['result'].append(label)
        except ValueError:
            print(f'Bad path: {yolo_path}')

        with open(os.path.join(STUDIO_LABEL_DIR, basename.replace('.jpg', '.json')), 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == '__main__':
    main()
