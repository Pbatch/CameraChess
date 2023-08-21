import os
from collections import Counter
from glob import glob

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from camera_chess.constants import YOLO_DIR, CLASSES


def main():
    bad_labels = []
    exempt_datasets = ['roboflow_9']
    piece_to_max_count = {'black-king': 1,
                          'white-king': 1,
                          'black-queen': 2,
                          'white-queen': 2,
                          'white-pawn': 8,
                          'black-pawn': 8,
                          'white-bishop': 2,
                          'black-bishop': 2,
                          'white-rook': 2,
                          'black-rook': 2,
                          'white-knight': 2,
                          'black-knight': 2}
    for label_path in tqdm(sorted(glob(os.path.join(YOLO_DIR, 'train', 'labels', '*.txt')))):
        with open(label_path) as f:
            lines = [line.strip() for line in f.readlines()]

        if not len(lines):
            continue

        classes = []
        bboxes = []
        for line in lines:
            cls, *bbox = line.split()
            classes.append(CLASSES[int(cls)])
            bboxes.append([float(i) for i in bbox])
        bboxes = np.array(bboxes)
        class_counts = dict(Counter(classes))

        bboxes[:, 0] -= bboxes[:, 2] / 2
        bboxes[:, 1] -= bboxes[:, 3] / 2
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes = torch.Tensor(bboxes)
        iou = torchvision.ops.box_iou(bboxes, bboxes)
        high_iou = torch.nonzero(torch.triu(iou, diagonal=1) > 0.9)

        basename = os.path.basename(label_path)
        for i, j in high_iou:
            bad_labels.append([basename, classes[i], classes[j], 'overlapping bboxes'])

        root_dataset = '_'.join(basename.split('_')[:-1])
        if root_dataset in exempt_datasets:
            continue

        bad_pieces = False
        bad_count = {}
        for piece in CLASSES:
            if class_counts.get(piece, 0) > piece_to_max_count[piece]:
                bad_count[piece] = class_counts[piece]
                bad_pieces = True
        if bad_pieces:
            bad_labels.append([basename, bad_count, 'bad pieces'])

    for label in bad_labels:
        print(label)


if __name__ == '__main__':
    main()
