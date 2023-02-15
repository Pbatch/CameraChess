import json
import os
import shutil

from imagesize import imagesize
from tqdm import tqdm

from camera_chess.constants import CLASSES
from glob import glob


def main():
    dataset_name = 'roboflow_4'
    for image_path in tqdm(glob(f'data/{dataset_name}/images/*.jpg')):
        yolo_path = image_path.replace('images', 'yolov5').replace('.jpg', '.txt')

        basename = os.path.basename(image_path)
        new_image_path = os.path.join('label_studio', 'files', 'images', basename)
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

        with open(os.path.join('label_studio', 'files', 'yolo', basename.replace('.jpg', '.json')), 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == '__main__':
    main()
