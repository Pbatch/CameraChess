import os
from collections import namedtuple

import torch

from camera_chess.constants import MODEL_DIR
from camera_chess.export.wrapped_model import load_model

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, model_basename, device='cuda:0'):
        self.model_basename = model_basename
        self.device = device

        self.model = load_model(model_path=os.path.join(MODEL_DIR, self.model_basename),
                                device=self.device)

    def run(self, images):
        images = torch.tensor(images, device=self.device)
        pred = self.model(images).detach().cpu().numpy()

        return pred
