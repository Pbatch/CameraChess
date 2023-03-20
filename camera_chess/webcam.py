import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class Webcam:
    def __init__(self, keypoints, fps=8, camera_id=1, padding=None):
        self.keypoints = keypoints
        self.fps = fps
        self.camera_id = camera_id
        self.padding = padding if padding is not None else [50, 100, 50, 10]

        self.roi = self._get_roi()
        self.l, self.t, self.r, self.b = [int(i) for i in self.roi]
        self.width = float(self.roi[2] - self.roi[0])
        self.height = float(self.roi[3] - self.roi[1])
        self.new_keypoints = self.keypoints - np.array([self.roi[0], self.roi[1]], dtype=np.float32)

        self.vid = None

    def _get_roi(self):
        roi = [max(np.min(self.keypoints[:, 0]) - self.padding[0], 0),
               max(np.min(self.keypoints[:, 1]) - self.padding[1], 0),
               np.max(self.keypoints[:, 0]) + self.padding[2],
               np.max(self.keypoints[:, 1]) + self.padding[3]]
        return roi

    def _load_vid(self):
        vid = cv2.VideoCapture(self.camera_id)
        while not vid.isOpened():
            time.sleep(1)
        return vid

    def save_start_image(self):
        self.vid = self._load_vid()
        while True:
            success, image = self.vid.read()
            if not success:
                continue
            image = image[..., ::-1]
            Image.fromarray(image).convert('RGB').save('start_image.jpg')
            return

    def __iter__(self):
        self.vid = self._load_vid()

        while not self.vid.isOpened():
            time.sleep(1)

        start = time.time()
        while True:
            success, image = self.vid.read()
            if not success:
                continue
            elapsed = time.time() - start
            if elapsed > 1 / self.fps:
                image = image[self.t:self.b, self.l:self.r, ::-1]
                yield image
                start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vid.release()
        cv2.destroyAllWindows()


def main():
    keypoints = np.array([[0, 0], [100, 0], [100, 0], [0, 100]], dtype=np.float32)
    webcam = Webcam(keypoints, camera_id=1)
    webcam.save_start_image()
    exit(1)
    for _ in tqdm(webcam):
        pass


if __name__ == '__main__':
    main()
