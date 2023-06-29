from collections import namedtuple

import numpy as np
from openvino.runtime import Core, Tensor

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, keypoints, model_path, weights_path=''):
        self.keypoints = keypoints
        self.model_path = model_path
        self.weights_path = weights_path

        model = Core().read_model(model=self.model_path,
                                  weights=self.weights_path)
        self.model = Core().compile_model(model=model, device_name="CPU",
                                          config={"PERFORMANCE_HINT": "THROUGHPUT"})
        self.input_layer_ir = self.model.input(0)
        self.infer_request = self.model.create_infer_request()

    def filter_by_roi(self, pred):
        piece_centers = np.vstack([(pred[:, 1] + pred[:, 3]) / 2,
                                   pred[:, 4] - ((pred[:, 3] - pred[:, 1]) / 4)]).T
        mask = np.ones(len(piece_centers), dtype=bool)
        for i in range(4):
            v1 = self.keypoints[i-1] - self.keypoints[i]
            v2 = piece_centers - self.keypoints[i]
            cross_products = np.cross(v1, v2)
            mask &= cross_products < 0
        pred = pred[mask]
        return pred

    def run(self, images):
        self.infer_request.set_tensor(self.input_layer_ir, Tensor(images))
        self.infer_request.infer()
        pred = self.infer_request.get_output_tensor(0).data
        pred = self.filter_by_roi(pred)
        pred = [pred[pred[:, 0] == i, 1:].tolist() for i in range(len(images))]

        return pred
