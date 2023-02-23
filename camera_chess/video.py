import os

import cv2
import numpy as np
from decord import VideoReader
from PIL import Image
from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE
from camera_chess.utils import warp


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
        self.l, self.t, self.r, self.b = [int(i) for i in self.roi]
        self.width = float(self.roi[2] - self.roi[0])
        self.height = float(self.roi[3] - self.roi[1])
        self.new_keypoints = self.video_config.keypoints - np.array([self.roi[0], self.roi[1]])
        self.frames = [i for i in range(len(self.vr))
                       if (self.start_frame <= i <= self.end_frame)
                       and i % self.mod == 0]

    def _get_roi(self):
        border = np.array([[-2, -2], [10, -2], [10, 10], [-2, 10]], dtype=np.float32) * SQUARE_SIZE
        extremities = warp(border, self.video_config.keypoints)

        height, width = self.vr[0].asnumpy().shape[:2]
        roi = [max(np.min(extremities[:, 0]), 0),
               max(np.min(extremities[:, 1]), 0),
               min(np.max(extremities[:, 0]), width),
               min(np.max(extremities[:, 1]), height)]
        return roi

    def __iter__(self):
        for frame in self.frames:
            image = self.vr[frame].asnumpy()
            image = image[self.t:self.b, self.l:self.r]
            yield image, frame

    def __len__(self):
        return len(self.frames)

    def save_start_image(self):
        image = self.vr[self.frames[0]].asnumpy()
        Image.fromarray(image).save('start_image.jpg')
