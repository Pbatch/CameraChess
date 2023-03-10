from collections import namedtuple

import numpy as np
from scipy.spatial import KDTree
from ultralytics.tracker import BYTETracker
from ultralytics.yolo.utils import IterableSimpleNamespace

from camera_chess.constants import CLASSES, SQUARE_SIZE
from camera_chess.utils import warp, get_square

Track = namedtuple("Track", "piece center square score bbox speed")


class Tracker:
    def __init__(self, fps, keypoints,
                 track_high_thresh=0.6,
                 track_low_thresh=0.1,
                 new_track_thresh=0.6,
                 track_buffer=60,
                 match_thresh=0.8):
        self.fps = fps
        self.keypoints = keypoints

        args = {'track_high_thresh': track_high_thresh,
                'track_low_thresh': track_low_thresh,
                'new_track_thresh': new_track_thresh,
                'track_buffer': track_buffer,
                'match_thresh': match_thresh}
        self.tracker = BYTETracker(IterableSimpleNamespace(**args),
                                   frame_rate=self.fps)

        grid = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        square_centers = warp(grid, self.keypoints)
        self.kd_tree = KDTree(square_centers)

    def update(self, detections):
        self.tracker.update(detections)
        tracks = []
        for strack in self.tracker.tracked_stracks:
            if not strack.is_activated:
                continue
            piece = CLASSES[strack.cls]
            bbox = strack.tlbr.tolist()
            _, _, _, _, vx, vy, _, _ = strack.mean
            speed = vx ** 2 + vy ** 2
            center = [(bbox[0] + bbox[2]) / 2,
                      bbox[3] - ((bbox[2] - bbox[0]) / 4)]
            square = get_square(self.kd_tree.query(center)[1])
            score = strack.score
            track = Track(piece, center, square, score, bbox, speed)
            tracks.append(track)
        return tracks
