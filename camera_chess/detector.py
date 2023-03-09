from collections import namedtuple

import numpy as np
import onnxruntime

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, model_path, keypoints, conf_thres=0.1, iou_thres=0.4):
        self.model_path = model_path
        self.keypoints = keypoints
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.sess = onnxruntime.InferenceSession(self.model_path)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def _filter_by_roi(self, pred):
        piece_centers = np.vstack([(pred[:, 0] + pred[:, 2]) / 2,
                                   pred[:, 3] - ((pred[:, 2] - pred[:, 0]) / 4)]).T
        mask = np.ones(len(piece_centers), dtype=bool)
        for i in range(4):
            v1 = self.keypoints[i-1] - self.keypoints[i]
            v2 = piece_centers - self.keypoints[i]
            cross_products = np.cross(v1, v2)
            mask &= cross_products < 0
        pred = pred[mask]
        return pred

    def run(self, image: np.ndarray):
        pred = self.sess.run([self.output_name], {self.input_name: np.expand_dims(image, axis=0)})[0]
        pred = self._filter_by_roi(pred)

        detections = Detections(pred[:, :4], pred[:, 4], pred[:, 5].astype(int))
        return detections
