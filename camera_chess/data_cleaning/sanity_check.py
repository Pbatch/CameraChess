import json
import os
from collections import Counter, defaultdict
from glob import glob

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from camera_chess.constants import PIECES_DIR, CLASSES


def main():
    bad_labels = defaultdict(lambda: defaultdict(list))
    exempt_datasets = ['roboflow_9', 'roboflow_public', 'roboflow_ppp']
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
    for label_path in tqdm(sorted(glob(os.path.join(PIECES_DIR, 'train', 'labels', '*.txt')))):
        with open(label_path) as f:
            lines = [line.strip() for line in f.readlines()]

        if not len(lines):
            continue

        basename = os.path.basename(label_path)
        *root_dataset, id_ = basename.split('_')
        root_dataset = '_'.join(root_dataset)
        id_ = int(id_.replace('.txt', '')) + 1

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
        high_iou = torch.nonzero(torch.triu(iou, diagonal=1) > 0.85)
        for i, j in high_iou:
            bad_labels[root_dataset][id_].append(['overlapping_bboxes', classes[i], classes[j]])

        bad_pieces = False
        bad_count = {}
        for piece in CLASSES:
            if class_counts.get(piece, 0) > piece_to_max_count[piece] and root_dataset not in exempt_datasets:
                bad_count[piece] = class_counts[piece]
                bad_pieces = True
        if bad_pieces:
            bad_labels[root_dataset][id_].append(['bad_pieces', bad_count])

        for bbox in bboxes.numpy():
            if bbox[2] < 0.01 or bbox[3] < 0.01:
                bad_labels[root_dataset][id_].append(['small_bbox', bbox.tolist()])

    bad_labels = dict({k: dict(v) for k, v in bad_labels.items()})
    with open("bad_labels.json", "w") as f:
        json.dump(bad_labels, f, indent=2)


if __name__ == '__main__':
    main()
