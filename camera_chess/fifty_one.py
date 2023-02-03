import json
import os

import fiftyone as fo

from camera_chess.constants import UPLOAD_DIR, LABEL_STUDIO_DIR


def main():
    with open(os.path.join(LABEL_STUDIO_DIR, 'data/export/project-1-at-2023-02-02-23-52-d5559bb1.json')) as f:
        d = json.load(f)
    labels = ['h1', 'a1', 'a8', 'h8']

    samples = []
    for i in d:
        filepath = os.path.join(UPLOAD_DIR, *i['data']['img'].split('/')[-2:])
        sample = fo.Sample(filepath=filepath)
        points = {}
        for annotation in i['annotations']:
            for entry in annotation['result']:
                value = entry['value']
                x = value['x'] / 100
                y = value['y'] / 100
                points[value['keypointlabels'][0]] = [x, y]
        keypoints = [fo.Keypoint(label="board",
                                 points=[points[s] for s in labels])]
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints)
        samples.append(sample)
    dataset = fo.Dataset()
    dataset.add_samples(samples)

    dataset.default_skeleton = fo.KeypointSkeleton(labels=labels, edges=[[0, 1, 2, 3, 0]])
    dataset.default_skeleton.labels[-1] = "board"
    dataset.save()

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    main()
