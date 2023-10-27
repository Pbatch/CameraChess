import os

import numpy as np
from PIL import Image
from decord import VideoReader


class Video:
    def __init__(self, video_config, target_fps=1, padding=None):
        self.video_config = video_config
        self.target_fps = target_fps
        self.padding = padding if padding is not None else [50, 100, 50, 10]

        if not os.path.isfile(self.video_config.path):
            raise FileNotFoundError(f'{self.video_config.path} does not exist')

        self.vr = VideoReader(self.video_config.path, num_threads=1)
        self.fps = self.vr.get_avg_fps()
        self.start_frame = self.video_config.start * self.fps if self.video_config.start is not None else 0
        self.end_frame = self.video_config.end * self.fps if self.video_config.end is not None else float('inf')
        self.mod = int(round(self.fps / self.target_fps))

        self.frames = [i for i in range(len(self.vr))
                       if (self.start_frame <= i <= self.end_frame)
                       and i % self.mod == 0]

        if self.video_config.keypoints is not None:
            self.roi = self._get_roi()
            self.l, self.t, self.r, self.b = [int(i) for i in self.roi]
            self.width = float(self.roi[2] - self.roi[0])
            self.height = float(self.roi[3] - self.roi[1])
            self.new_keypoints = self.video_config.keypoints - np.array([self.roi[0], self.roi[1]], dtype=np.float32)
        else:
            self.height, self.width = self.vr[0].asnumpy().shape[:2]
            self.l, self.t, self.r, self.b = 0, 0, self.width, self.height
            self.new_keypoints = None

    def _get_roi(self):
        height, width = self.vr[0].asnumpy().shape[:2]
        roi = [max(np.min(self.video_config.keypoints[:, 0]) - self.padding[0], 0),
               max(np.min(self.video_config.keypoints[:, 1]) - self.padding[1], 0),
               min(np.max(self.video_config.keypoints[:, 0]) + self.padding[2], width),
               min(np.max(self.video_config.keypoints[:, 1]) + self.padding[3], height)]
        return roi

    def __iter__(self):
        for frame in self.frames:
            image = self.vr[frame].asnumpy()
            image = image[self.t:self.b, self.l:self.r]
            yield image, frame

    def __len__(self):
        return len(self.frames)

    def save_start_image(self):
        image = self.vr[self.frames[50]].asnumpy()
        Image.fromarray(image).save('start_image.jpg')
