import json
import os
from glob import glob

import fiftyone as fo

from camera_chess.constants import DATA_DIR


def load_label_studio():
    samples = []
    keypoint_order = ['h1', 'a1', 'a8', 'h8']
    for dataset in ['peter/kasparov_immortal']:
        for image_path in glob(os.path.join(DATA_DIR, dataset, 'images', '*')):
            sample = fo.Sample(filepath=image_path)

            label_path = image_path.replace('images', 'labels').replace('.jpg', '.json')
            with open(label_path, 'r') as f:
                label = json.load(f)
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
    dataset.save()
    return dataset


def load_yolo():
    dataset = fo.Dataset()
    for split in ['train', 'val']:
        dataset.add_dir(
            dataset_dir=os.path.join(DATA_DIR, 'yolo'),
            yaml_path=os.path.join(DATA_DIR, 'yolo/data.yaml'),
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            tags=split,
        )
    return dataset


def main():
    # dataset = load_yolo().shuffle()
    dataset = load_label_studio()
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    main()
