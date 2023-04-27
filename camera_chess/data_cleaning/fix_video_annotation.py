import json
import os
from copy import deepcopy

import chess
import numpy as np
from PIL import Image
from scipy.spatial import KDTree

from camera_chess.constants import DATA_DIR, SQUARE_SIZE, CORNERS
from camera_chess.utils import warp, get_square, load_video_config
from camera_chess.visualizer import Visualizer


def main():
    export_id = "project-1-at-2023-04-04-15-47-895c61ad"
    dataset = 'youtube/bobby_fischer'
    verbose = False

    video_config = load_video_config(dataset)

    visualizer = Visualizer()
    board = chess.Board(fen=video_config.fen)
    with open(os.path.join('label_studio', 'data', 'export', f'{export_id}.json')) as f:
        labels = json.load(f)

    os.makedirs(os.path.join(DATA_DIR, dataset, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, dataset, 'images'), exist_ok=True)
    new_label = None
    for i, label in enumerate(labels):
        board.push(chess.Move.from_uci(video_config.moves[i]))

        old_label = new_label
        if old_label is None:
            new_label = {'keypoints': {s: [] for s in ['h1', 'a1', 'a8', 'h8']},
                         'bboxes': []}
        else:
            new_label = deepcopy(old_label)
            new_label['bboxes'] = []

        keypoint_annotations = [a for a in label['annotations'][0]['result'] if a['type'] == 'keypointlabels']
        bbox_annotations = [a for a in label['annotations'][0]['result'] if a['type'] == 'rectanglelabels']
        for a in keypoint_annotations:
            value = a['value']
            key = value['keypointlabels'][0]
            x = float(value['x']) / 100
            y = float(value['y']) / 100
            new_label['keypoints'][key] = [x, y]
        grid = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        keypoints = np.array([new_label['keypoints'][key] for key in CORNERS], dtype=np.float32)
        square_centers = warp(grid, keypoints)
        kd_tree = KDTree(square_centers)

        for a in bbox_annotations:
            value = a['value']
            class_ = value['rectanglelabels'][0]
            bbox = [class_] + [float(value[s]) / 100 for s in ['x', 'y', 'width', 'height']]
            new_label['bboxes'].append(bbox)

        if old_label is not None:
            for bbox in old_label['bboxes']:
                l, t, r, b = [bbox[1], bbox[2], bbox[1] + bbox[3], bbox[2] + bbox[4]]
                center = [(l + r) / 2,
                          b - ((r - l) / 4)]
                square = get_square(kd_tree.query(center)[1])
                if board.piece_at(chess.parse_square(square)) is not None:
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
            input()


if __name__ == '__main__':
    main()
