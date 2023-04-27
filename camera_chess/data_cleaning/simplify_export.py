import json
import os
import time

from PIL import Image

from camera_chess.constants import DATA_DIR
from camera_chess.visualizer import Visualizer


def main():
    export_id = 'project-1-at-2023-04-14-15-04-5564142b'
    dataset = 'roboflow/7'
    verbose = False
    visualizer = Visualizer()
    with open(os.path.join('label_studio', 'data', 'export', f'{export_id}.json')) as f:
        labels = json.load(f)

    os.makedirs(os.path.join(DATA_DIR, dataset, 'labels'))
    os.makedirs(os.path.join(DATA_DIR, dataset, 'images'))
    for i, label in enumerate(labels):
        new_label = {'keypoints': {s: [] for s in ['h1', 'a1', 'a8', 'h8']},
                     'bboxes': []}
        for annotation in label['annotations'][0]['result']:
            value = annotation['value']
            type_ = annotation['type']

            if type_ == 'keypointlabels':
                key = value['keypointlabels'][0]
                x = float(value['x']) / 100
                y = float(value['y']) / 100
                new_label['keypoints'][key] = [x, y]
            elif type_ == 'rectanglelabels':
                class_ = value['rectanglelabels'][0]
                bbox = [class_] + [float(value[s]) / 100 for s in ['x', 'y', 'width', 'height']]
                new_label['bboxes'].append(bbox)

        if all([len(v) == 0 for v in new_label['keypoints'].values()]):
            new_label.pop('keypoints')
        new_label_path = f'data/{dataset}/labels/{i}.json'
        with open(new_label_path, 'w') as f:
            json.dump(new_label, f, indent=4)

        basename = os.path.basename(label['data']['img'])
        image = Image.open(os.path.join('label_studio', 'files', 'images', basename))
        image = image.convert('RGB')
        new_image_path = f'data/{dataset}/images/{i}.jpg'
        image.save(new_image_path)

        if verbose:
            image = Visualizer.add_bboxes(image, new_label['bboxes'])
            image.show()
            time.sleep(1)


if __name__ == '__main__':
    main()
