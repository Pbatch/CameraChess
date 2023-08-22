from collections import namedtuple

import numpy as np
import torch

from camera_chess.export.wrapped_model import load_model

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, keypoints, model_path, device='cpu'):
        self.keypoints = keypoints
        self.model_path = model_path
        self.device = device

        self.model = load_model(self.model_path, device=self.device)

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
        images = torch.tensor(images, device=self.device)
        pred = self.model(images).detach().cpu().numpy()
        pred = self.filter_by_roi(pred)
        pred = [pred[pred[:, 0] == i, 1:].tolist() for i in range(len(images))]

        return pred
