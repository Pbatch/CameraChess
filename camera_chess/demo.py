import argparse
import json
import sys

import numpy as np
import requests
from tqdm import tqdm

from camera_chess.detector import Detector
from camera_chess.utils import load_video_config, clear_dir
from camera_chess.utils import serialize, deserialize
from camera_chess.video import Video

# Need for deserialization
sys.path.insert(0, '../CameraChessWeb/aws/tracker')
from state import State
from tracker import Tracker


def aws(preds, tracker, state, tracker_kwargs, state_kwargs):
    d = {'preds': preds}
    if tracker is None or state is None:
        d['tracker_kwargs'] = tracker_kwargs
        d['state_kwargs'] = state_kwargs
    else:
        d['tracker'] = serialize(tracker)
        d['state'] = serialize(state)
    response = requests.post("https://zht7adkgfc.execute-api.eu-west-2.amazonaws.com/prod/tracker",
                             json=d)

    if response.status_code != 200:
        print(response)
        exit(1)

    content = json.loads(response.content)
    tracker = deserialize(content['tracker'])
    state = deserialize(content['state'])

    return tracker, state


def local(preds, tracker, state, tracker_kwargs, state_kwargs):
    if tracker is None:
        tracker = Tracker(**tracker_kwargs)

    if state is None:
        state = State(**state_kwargs)

    for pred in preds:
        pred = np.array(pred) if len(pred) else np.empty(shape=(0, 6))
        tracker.update(pred)
        state.update(tracker.tracks)
    return tracker, state


def main(dataset):
    batch_size = 8
    video_config = load_video_config(dataset)
    video = Video(video_config, target_fps=8)
    video.save_start_image()
    detector = Detector(model_path='models/480S-quant.xml',
                        weights_path='models/480S-quant.bin',
                        keypoints=video.new_keypoints)
    tracker_kwargs = {'fps': video.target_fps,
                      'keypoints': video.new_keypoints.tolist(),
                      'new_track_thresh': 0.3,
                      'track_high_thresh': 0.3,
                      'track_low_thresh': 0.1}
    state_kwargs = {'fen': video_config.fen}
    clear_dir('debug')

    tracker = None
    state = None
    images = np.empty((batch_size, int(video.height), int(video.width), 3), dtype=np.uint8)
    i = 0
    for image, frame in tqdm(video, desc='Frame'):
        images[i] = image
        i += 1

        if i != batch_size and frame != video.frames[-1]:
            continue

        preds = detector.run(images[:i])
        tracker, state = local(preds, tracker, state, tracker_kwargs, state_kwargs)
        i = 0

    print(state.game)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)

