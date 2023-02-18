import cv2
import numpy as np
from PIL import Image

from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE


class Video:
    def __init__(self, video_path, keypoints, start=None, end=None, target_fps=1):
        self.video_path = video_path
        self.keypoints = keypoints
        self.start = start
        self.end = end
        self.target_fps = target_fps

        self.roi = self._get_roi()
        self.new_keypoints = self.keypoints - np.array([self.roi[0], self.roi[1]])

    def _get_roi(self):
        target = np.array([[BOARD_SIZE, BOARD_SIZE],
                           [0, BOARD_SIZE],
                           [0, 0],
                           [BOARD_SIZE, 0]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(self.keypoints, target)
        inv_matrix = np.linalg.inv(matrix)
        warped_extremities = np.array([[[-1, -1], [9, -1], [9, 9], [-1, 9]]], dtype=np.float32) * SQUARE_SIZE
        extremities = cv2.perspectiveTransform(warped_extremities, inv_matrix)[0]

        roi = [np.min(extremities[:, 0]),
               np.min(extremities[:, 1]),
               np.max(extremities[:, 0]),
               np.max(extremities[:, 1])]
        return roi

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        start_frame = self.start * fps if self.start is not None else 0
        end_frame = self.end * fps if self.end is not None else float('inf')
        mod = int(round(fps / self.target_fps))
        i = 0
        while True:
            success = cap.grab()
            if not success:
                break

            if i % mod == 0 and start_frame <= i <= end_frame:
                _, image = cap.retrieve()
                image = Image.fromarray(image[..., ::-1]).convert('RGB')
                image = image.crop(self.roi)
                yield image
            i += 1
