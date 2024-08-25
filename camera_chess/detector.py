import os
from collections import namedtuple

import cv2
import numpy as np
import onnxruntime as ort

from camera_chess.constants import MODEL_DIR, CORNERS
from camera_chess.utils import get_roi

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, model_basename, fill_colour=114, device='cuda:0'):
        self.model_basename = model_basename
        self.fill_colour = fill_colour
        self.device = device

        providers = [('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"})]
        self.sess = ort.InferenceSession(os.path.join(MODEL_DIR, self.model_basename),
                                         providers=providers)
        self.inputs = self.sess.get_inputs()[0]
        self.outputs = self.sess.get_outputs()[0]
        self.io_binding = self.sess.io_binding()
        self.io_binding.bind_output(self.outputs.name)

        # shape should be B, C, H, W
        self.model_height = self.inputs.shape[2]
        self.model_width = self.inputs.shape[3]
        self.desired_ratio = self.model_height / self.model_width

        self.v10 = "v10" in self.model_basename

    def _resize(self, x):
        height, width = x.shape[:2]
        ratio = height / width
        if ratio > self.desired_ratio:
            height = self.model_height
            width = int(self.model_height / ratio)
        else:
            width = self.model_width
            height = int(self.model_width * ratio)

        x = cv2.resize(x, (width, height))

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
        if not self.v10:
            y = y.T
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

    def run(self, image, keypoints=None):
        height, width = image.shape[:2]
        if keypoints is not None:
            # Crop out region of interest
            keypoints_arr = np.array([keypoints[s].tolist() for s in CORNERS])
            roi = get_roi(keypoints_arr, width, height, self.model_width, self.model_height)
        else:
            roi = [0, 0, width, height]
        image = image[roi[1]:roi[3], roi[0]:roi[2]]

        # Resize, pad and scale
        image = self._resize(image)
        image, padding = self._pad(image)
        image = image.astype(np.float16) / 255
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # Inference
        self.io_binding.bind_cpu_input(self.inputs.name, image)
        self.sess.run_with_iobinding(self.io_binding)
        y = self.io_binding.copy_outputs_to_cpu()[0][0].astype(np.float32)

        # Rescale bboxes to match original image size
        y = self._fix_bboxes(y, roi, padding)

        return y, roi
