import os

import numpy as np
from PIL import Image
from decord import VideoReader


class Video:
    def __init__(self, video_config, target_fps=1, model_width=640, model_height=384):
        self.video_config = video_config
        self.target_fps = target_fps
        self.model_width = model_width
        self.model_height = model_height

        self.desired_ratio = self.model_height / self.model_width

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
        elif self.video_config.roi is not None:
            self.l, self.t, self.r, self.b = [int(i) for i in video_config.roi]
            self.width = float(self.r - self.l)
            self.height = float(self.b - self.t)
            self.new_keypoints = None
        else:
            self.height, self.width = self.vr[0].asnumpy().shape[:2]
            self.l, self.t, self.r, self.b = 0, 0, self.width, self.height
            self.new_keypoints = None
        # self.height, self.width = self.vr[0].asnumpy().shape[:2]
        # self.l, self.t, self.r, self.b = 0, 0, self.width, self.height
        # self.new_keypoints = self.video_config.keypoints

    def _get_roi(self):
        height, width = self.vr[0].asnumpy().shape[:2]
        x_min = np.min(self.video_config.keypoints[:, 0])
        x_max = np.max(self.video_config.keypoints[:, 0])
        y_min = np.min(self.video_config.keypoints[:, 1])
        y_max = np.max(self.video_config.keypoints[:, 1])

        roi_width = x_max - x_min
        roi_height = y_max - y_min
        padding_left = roi_width // 16
        padding_right = roi_width // 16
        padding_top = roi_height // 16
        padding_bottom = roi_height // 16

        padded_roi_width = roi_width + padding_left + padding_right
        padded_roi_height = roi_height + padding_top + padding_bottom
        ratio = padded_roi_height / padded_roi_width

        if ratio > self.desired_ratio:
            target_width = padded_roi_height / self.desired_ratio
            height = self.model_height
            width = self.model_height / self.desired_ratio
            dx = target_width - padded_roi_width
            padding_left += dx // 2
            padding_right += dx - (dx // 2)
        else:
            target_height = padded_roi_width * self.desired_ratio
            padding_top += target_height - padded_roi_height

        roi = [max(x_min - padding_left, 0),
               max(y_min - padding_top, 0),
               min(x_max + padding_right, width),
               min(y_max + padding_bottom, height)]
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
