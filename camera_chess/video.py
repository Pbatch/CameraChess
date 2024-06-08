import os

from PIL import Image
from decord import VideoReader


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
        self.height, self.width = self.vr[0].asnumpy().shape[:2]
        self.video_id = os.path.splitext(os.path.basename(self.video_config.path))[0].replace('.', '_')

        self.frames = [i for i in range(len(self.vr))
                       if (self.start_frame <= i <= self.end_frame)
                       and i % self.mod == 0]

    def __iter__(self):
        for frame in self.frames:
            image = self.vr[frame].asnumpy()
            yield image, frame

    def __len__(self):
        return len(self.frames)

    def save_start_image(self):
        image = self.vr[self.frames[500]].asnumpy()
        Image.fromarray(image).save('start_image.jpg')
