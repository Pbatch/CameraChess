import os

import cv2
import numpy as np
from decord import VideoReader

from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE


class Video:
    def __init__(self, video_config, target_fps=1):
        self.video_config = video_config
        self.target_fps = target_fps

        if not os.path.isfile(self.video_config.path):
            raise FileNotFoundError(f'{self.video_config.path} does not exist')

        self.vr = VideoReader(self.video_config.path, num_threads=1)
        self.fps = self.vr.get_avg_fps()
        self.start_frame = self.video_config.start * self.fps if self.video_config.start is not None else 0
        self.end_frame = self.video_config.end * self.fps if self.video_config.end is not None else float('inf')
        self.mod = int(round(self.fps / self.target_fps))

        self.roi = self._get_roi()
        self.width = float(self.roi[2] - self.roi[0])
        self.height = float(self.roi[3] - self.roi[1])
        self.new_keypoints = self.video_config.keypoints - np.array([self.roi[0], self.roi[1]])

    def _get_roi(self):
        target = np.array([[BOARD_SIZE, BOARD_SIZE],
                           [0, BOARD_SIZE],
                           [0, 0],
                           [BOARD_SIZE, 0]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(self.video_config.keypoints, target)
        inv_matrix = np.linalg.inv(matrix)
        warped_extremities = np.array([[[-1, -1], [9, -1], [9, 9], [-1, 9]]], dtype=np.float32) * SQUARE_SIZE
        extremities = cv2.perspectiveTransform(warped_extremities, inv_matrix)[0]

        roi = [np.min(extremities[:, 0]),
               np.min(extremities[:, 1]),
               np.max(extremities[:, 0]),
               np.max(extremities[:, 1])]
        return roi

    def __iter__(self):
        frames = [i for i in range(len(self.vr))
                  if (self.start_frame <= i <= self.end_frame) and i % self.mod == 0]
        for frame in frames:
            image = self.vr[frame].asnumpy()
            l, t, r, b = [int(i) for i in self.roi]
            image = image[t:b, l:r]
            yield image, frame
