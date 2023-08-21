import json
import os
from glob import glob

import fiftyone as fo

from camera_chess.constants import DATA_DIR, CLASSES


def load_label_studio():
    samples = []
    keypoint_order = ['h1', 'a1', 'a8', 'h8']
    for dataset in ['peter_wooden/wells_shirov']:
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
    dataset.save()
    return dataset


def load_yolo():
    samples = []
    for split in ['train', 'val']:
        for label_path in glob(os.path.join(DATA_DIR, 'yolo', split, 'labels', '*.txt')):
            image_path = label_path.replace('labels', 'images', 1).replace('.txt', '.jpg')
            source_dataset = '_'.join(os.path.basename(label_path).split('_')[:-1])
            sample = fo.Sample(filepath=image_path,
                               tags=[split, source_dataset],
                               basename=os.path.basename(label_path))
            with open(label_path) as f:
                lines = [line.strip() for line in f.readlines()]

            detections = []
            for line in lines:
                cls_idx, xc, yc, w, h = [float(i) for i in line.split()]
                label = CLASSES[int(cls_idx)]
                bounding_box = [xc - w/2, yc - h/2, w, h]
                detection = fo.Detection(label=label,
                                         bounding_box=bounding_box,
                                         tags=[label])
                detections.append(detection)
            sample['groundtruth'] = fo.Detections(detections=detections)
            samples.append(sample)

    dataset = fo.Dataset()
    dataset.add_samples(samples)
    dataset.save()

    return dataset


def main():
    dataset = load_yolo().shuffle()
    # dataset = load_label_studio().shuffle()
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    main()
