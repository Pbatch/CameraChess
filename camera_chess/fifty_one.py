import argparse
import json
import os
from glob import glob

import fiftyone as fo
from tqdm import tqdm
import numpy as np

from camera_chess.constants import DATA_DIR, CLASSES, CORNERS
from camera_chess.export.wrapped_model import load_model
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import Compose, PILToTensor
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image


def load_label_studio():
    samples = []
    keypoint_order = ['h1', 'a1', 'a8', 'h8']
    for dataset in ['peter/random']:
        for image_path in glob(os.path.join(DATA_DIR, dataset, 'images', '*')):
            sample = fo.Sample(filepath=image_path)

            label_path = image_path.replace('images', 'labels').replace('.jpg', '.json')
            with open(label_path, 'r') as f:
                label = json.load(f)
            if 'keypoints' in label:
                keypoints = [fo.Keypoint(label="board",
                                         points=[label['keypoints'][s] for s in keypoint_order])]
                sample["keypoints"] = fo.Keypoints(keypoints=keypoints)

            detections = [fo.Detection(label=label, bounding_box=bbox)
                          for label, *bbox in label['bboxes']]
            sample["groundtruth"] = fo.Detections(detections=detections)
            samples.append(sample)

    dataset = fo.Dataset()
    dataset.add_samples(samples)

    dataset.default_skeleton = fo.KeypointSkeleton(labels=keypoint_order, edges=[[0, 1, 2, 3, 0]])
    dataset.default_skeleton.labels[-1] = "board"
    dataset = dataset.shuffle()
    dataset.save()
    return dataset


def load_yolo(pred_dir):
    samples = []
    for split in ['train', 'val']:
        paths = glob(os.path.join(DATA_DIR, 'yolo', split, 'labels', '*.txt'))
        for label_path in tqdm(sorted(paths)):
            image_path = label_path.replace('labels', 'images', 1).replace('.txt', '.jpg')
            source_dataset = '_'.join(os.path.basename(label_path).split('_')[:-1])
            sample = fo.Sample(filepath=image_path,
                               tags=[split, source_dataset],
                               basename=os.path.basename(label_path))
            with open(label_path) as f:
                lines = [line.strip() for line in f.readlines()]

            detections = []
            gt_boxes = []
            gt_labels = []
            for line in lines:
                cls_idx, xc, yc, w, h = [float(i) for i in line.split()]
                cls_idx = int(cls_idx)

                label = CLASSES[cls_idx]
                bounding_box = [xc - w / 2, yc - h / 2, w, h]
                detection = fo.Detection(label=label,
                                         bounding_box=bounding_box,
                                         tags=[label])
                detections.append(detection)

                gt_boxes.append(bounding_box)
                gt_labels.append(cls_idx)
            sample['groundtruth'] = fo.Detections(detections=detections)

            if pred_dir is None:
                sample['ap'] = 1.0
                samples.append(sample)
                continue

            predictions_path = os.path.join(pred_dir, os.path.basename(label_path))

            boxes = []
            scores = []
            labels = []
            predictions = []
            if os.path.isfile(predictions_path):
                with open(predictions_path) as f:
                    lines = [line.strip() for line in f.readlines()]
                for line in lines:
                    cls_idx, x, y, w, h, conf = [float(i) for i in line.split()]
                    cls_idx = int(cls_idx)
                    label = CLASSES[cls_idx]

                    bounding_box = [x - w / 2, y - h / 2, w, h]

                    boxes.append(bounding_box)
                    labels.append(cls_idx)
                    scores.append(conf)

                    prediction = fo.Detection(label=label,
                                              bounding_box=bounding_box,
                                              tags=[label])
                    predictions.append(prediction)
            sample['predictions'] = fo.Detections(detections=predictions)

            metric = MeanAveragePrecision(box_format="xywh")
            preds = [dict(boxes=torch.tensor(boxes),
                          scores=torch.tensor(scores),
                          labels=torch.tensor(labels))]
            target = [dict(boxes=torch.tensor(gt_boxes),
                           labels=torch.tensor(gt_labels))]
            metric.update(preds, target)
            result = metric.compute()
            ap = result['map'].item()
            if ap != -1.0:
                sample['ap'] = ap
            else:
                sample['ap'] = 1.0

            samples.append(sample)

    dataset = fo.Dataset()
    dataset.add_samples(samples)
    dataset = dataset.sort_by('ap')
    dataset.save()

    return dataset


def load_corners():
    samples = []
    for split in ['train', 'val']:
        paths = glob(os.path.join(DATA_DIR, 'corners', split, 'labels', '*.txt'))
        for label_path in tqdm(sorted(paths)):
            image_path = label_path.replace('labels', 'images', 1).replace('.txt', '.jpg')
            source_dataset = '_'.join(os.path.basename(label_path).split('_')[:-1])
            sample = fo.Sample(filepath=image_path,
                               tags=[split, source_dataset],
                               basename=os.path.basename(label_path))
            with open(label_path) as f:
                lines = [line.strip() for line in f.readlines()]

            detections = []
            gt_boxes = []
            gt_labels = []
            for line in lines:
                cls_idx, xc, yc, w, h = [float(i) for i in line.split()]
                cls_idx = int(cls_idx)

                label = CORNERS[cls_idx]
                bounding_box = [xc - w / 2, yc - h / 2, w, h]
                detection = fo.Detection(label=label,
                                         bounding_box=bounding_box,
                                         tags=[label])
                detections.append(detection)

                gt_boxes.append(bounding_box)
                gt_labels.append(cls_idx)
            sample['groundtruth'] = fo.Detections(detections=detections)
            samples.append(sample)

    dataset = fo.Dataset()
    dataset.add_samples(samples)
    dataset.save()

    return dataset


def main(mode, detections):
    if mode == 'yolo':
        dataset = load_yolo(detections)
    elif mode == 'corners':
        dataset = load_corners()
    else:
        # mode = 'studio'
        dataset = load_label_studio()
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, required=True, choices=['yolo', 'studio', 'corners'])
    parser.add_argument('--pred_dir', '-p', type=str, default=None)
    args = parser.parse_args()
    main(args.mode, args.pred_dir)
