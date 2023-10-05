from collections import namedtuple

import torch

from camera_chess.export.wrapped_model import load_model

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device

        self.model = load_model(self.model_path, device=self.device)

    def run(self, images):
        images = torch.tensor(images, device=self.device)
        pred = self.model(images).detach().cpu().numpy()

        return pred
