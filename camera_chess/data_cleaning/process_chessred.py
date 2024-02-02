import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.constants import DATA_DIR
from camera_chess.utils import clear_dir


def main():
    chessred_dir = os.path.join(DATA_DIR, 'chessred2k')
    clear_dir(os.path.join(chessred_dir, 'images'))
    clear_dir(os.path.join(chessred_dir, 'labels'))
    with open(os.path.join(chessred_dir, 'annotations.json'), 'r') as f:
        annotations = json.load(f)

    id_to_category = {d['id']: d['name'] for d in annotations['categories']}
    image_id_to_data = defaultdict(dict)

    for d in annotations['images']:
        image_id_to_data[d['id']]['width'] = d['width']
        image_id_to_data[d['id']]['height'] = d['height']
        image_id_to_data[d['id']]['path'] = d['path']
    for d in annotations['annotations']['pieces']:
        image_id = d['image_id']
        data = image_id_to_data[image_id]
        label = id_to_category[d['category_id']]
        try:
            x, y, w, h = d['bbox']
        except KeyError:
            continue
        if 'bboxes' not in image_id_to_data[image_id]:
            image_id_to_data[image_id]['bboxes'] = []
        image_id_to_data[image_id]['bboxes'].append([label,
                                                     x / data['width'],
                                                     y / data['height'],
                                                     w / data['width'],
                                                     h / data['height']])

    corners_map = {'bottom_right': 'h1', 'top_right': 'h8', 'top_left': 'a8', 'bottom_left': 'a1'}
    for d in annotations['annotations']['corners']:
        image_id = d['image_id']

        data = image_id_to_data[image_id]
        corners = np.array(list(d['corners'].values())) / [data['width'], data['height']]

        keypoints = {corners_map[k]: corner.tolist() for k, corner in zip(d['corners'].keys(), corners)}
        image_id_to_data[image_id]['keypoints'] = keypoints

    for k, v in tqdm(image_id_to_data.items()):
        if 'bboxes' not in v.keys():
            continue
        old_image_path = os.path.join(chessred_dir, v['path'].replace('images', 'raw_images'))
        new_image_path = os.path.join(chessred_dir, 'images', f'{k}.jpg')
        image = Image.open(old_image_path).convert('RGB')
        image.save(new_image_path)

        new_label_path = os.path.join(chessred_dir, 'labels', f'{k}.json')
        label = {'keypoints': v['keypoints'],
                 'bboxes': v['bboxes']}
        with open(new_label_path, 'w') as f:
            json.dump(label, f, indent=4)


if __name__ == '__main__':
    main()
