import os
from collections import namedtuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from camera_chess.constants import MODEL_DIR, CORNERS
from camera_chess.utils import get_roi

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, model_basename, model_width=640, model_height=384, fill_colour=114, device='cuda:0'):
        self.model_basename = model_basename
        self.model_width = model_width
        self.model_height = model_height
        self.fill_colour = fill_colour
        self.device = device

        self.sess = ort.InferenceSession(os.path.join(MODEL_DIR, self.model_basename),
                                         providers=['CUDAExecutionProvider'])
        self.desired_ratio = self.model_height / self.model_width

    def _resize(self, x):
        height, width = x.shape[:2]
        ratio = height / width
        if ratio > self.desired_ratio:
            height = self.model_height
            width = int(self.model_height / ratio)
        else:
            width = self.model_width
            height = int(self.model_width * ratio)

        x = np.array(Image.fromarray(x).resize((width, height)))

        return x

    def _pad(self, x):
        height, width = x.shape[:2]
        dx = self.model_width - width
        dy = self.model_height - height
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        padding = [pad_left, pad_right, pad_top, pad_bottom]
        x = np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                   mode="constant",
                   constant_values=self.fill_colour)

        return x, padding

    def _fix_bboxes(self, y, roi, padding):
        # xywh -> xyxy
        y[..., 0] -= y[..., 2] / 2
        y[..., 1] -= y[..., 3] / 2
        y[..., 2] += y[..., 0]
        y[..., 3] += y[..., 1]

        # Rescale predictions to original image
        roi_width = roi[2] - roi[0]
        roi_height = roi[3] - roi[1]
        y[..., [0, 2]] -= padding[0]
        y[..., [1, 3]] -= padding[2]
        y[..., [0, 2]] *= roi_width / (self.model_width - padding[0] - padding[1])
        y[..., [1, 3]] *= roi_height / (self.model_height - padding[2] - padding[3])

        y[..., [0, 2]] += roi[0]
        y[..., [1, 3]] += roi[1]

        return y

    def run(self, image, keypoints):
        # Crop out region of interest
        height, width = image.shape[:2]
        keypoints_arr = np.array([keypoints[s].tolist() for s in CORNERS])
        roi = get_roi(keypoints_arr, width, height, self.model_width, self.model_height)
        image = image[roi[1]:roi[3], roi[0]:roi[2]]

        # Resize, pad and scale
        image = self._resize(image)
        image, padding = self._pad(image)
        image = image.astype(np.float32) / 255
        image = np.transpose(image, (2, 0, 1))

        # Inference
        y = self.sess.run(None, {'images': np.expand_dims(image, 0)})[0][0]
        y = np.transpose(y, (1, 0))

        # Rescale bboxes to match original image size
        y = self._fix_bboxes(y, roi, padding)

        return y
