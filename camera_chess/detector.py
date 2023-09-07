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

    def run(self, images):
        images = torch.tensor(images, device=self.device)
        pred = self.model(images).detach().cpu().numpy()
        pred = [pred[pred[:, 0] == i, 1:].tolist() for i in range(len(images))]

        return pred
