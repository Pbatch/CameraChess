import json
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.board_detector import BoardDetector
from camera_chess.constants import DATA_DIR, CORNERS, CLASSES
from camera_chess.detector import Detector
from camera_chess.utils import video_config
from camera_chess.video import Video


def main(conf_threshold=0.1):
    detector = Detector(model_basename="640X_v10_pieces_640x448.onnx")
    board_detector = BoardDetector()

    autolabel_dir = os.path.join(DATA_DIR, "autolabel")
    image_dir = os.path.join(autolabel_dir, "images")
    label_dir = os.path.join(autolabel_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    video_paths = sorted(glob(os.path.join(DATA_DIR, "autolabel_videos", "*")))
    seen = set()
    for path in tqdm(video_paths):
        config = video_config(start=None, end=None, url=None, path=path, keypoints=None, fen=None, moves=None, roi=None)
        video = Video(video_config=config, target_fps=0.01)
        if video.video_id in seen:
            continue

        seen.add(video.video_id)
        for image, frame in video:
            corners, xcorners = board_detector.find_corners(image)

            if xcorners is None:
                continue

            if len(xcorners) < 20:
                continue

            if any([
                np.max(corners[:, 0]) >= video.width,
                np.max(corners[:, 1]) >= video.height,
                np.min(corners[:, 0]) <= 0,
                np.min(corners[:, 1]) <= 0
            ]):
                continue

            # Order doesn't matter
            keypoints = {i: j for i, j in zip(CORNERS, corners)}
            preds, roi = detector.run(image, keypoints)
            preds = preds[preds[:, 4] > conf_threshold]
            if len(preds) > 32:
                continue

            image = image[roi[1]: roi[3], roi[0]: roi[2]]
            frame_id = f'{video.video_id}_frame={frame}'
            pil_image = Image.fromarray(image).convert('RGB')
            pil_image.save(os.path.join(image_dir, f'{frame_id}.jpg'))

            preds[:, [0, 2]] = (preds[:, [0, 2]] - roi[0]) / pil_image.width
            preds[:, [1, 3]] = (preds[:, [1, 3]] - roi[1]) / pil_image.height
            preds[:, [2, 3]] -= preds[:, [0, 1]]
            bboxes = [[CLASSES[int(p[5])], *p[:4].tolist()] for p in preds]

            label = {"bboxes": bboxes}
            with open(os.path.join(label_dir, f'{frame_id}.json'), 'w') as f:
                json.dump(label, f, indent=4)
            break


if __name__ == '__main__':
    main()
