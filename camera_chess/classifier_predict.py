import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from camera_chess.classifier import ChessDataset
from camera_chess.classifier import load_model
from camera_chess.constants import DATA_DIR, MARGIN, SQUARE_SIZE, CLASSIFIER_CLASSES
from camera_chess.data_cleaning.make_classifier_data import warp_chessboard_image


class MobileDetector:
    def __init__(self, checkpoint_path, device='cpu'):
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.model = load_model(checkpoint_path=self.checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        self.batch = torch.empty((64, 3, 224, 224), device=self.device)

    def fill_batch(self, image, keypoints):
        pair_sums = [keypoints[-i][1] + keypoints[-i + 1][1] for i in range(4)]
        shift = np.argmin(pair_sums)
        keypoints = np.roll(keypoints, shift, axis=0)
        warped_image = warp_chessboard_image(np.array(image), keypoints)

        for i in range(64):
            x = i // 8
            y = i % 8
            # Clockwise rotation of 90 degrees
            for _ in range(shift):
                x, y = 7 - y, x

            x1 = int(MARGIN + SQUARE_SIZE * (x - 1))
            x2 = int(MARGIN + SQUARE_SIZE * (x + 2))
            y1 = int(MARGIN + SQUARE_SIZE * (y - 2))
            y2 = int(MARGIN + SQUARE_SIZE * (y + 1))
            crop = Image.fromarray(warped_image[y1:y2, x1:x2])
            self.batch[i] = ChessDataset.transform(crop)

    def run(self, image, keypoints):
        self.fill_batch(image, keypoints)
        res = self.model(self.batch)
        probs = nn.functional.softmax(res, dim=1).detach().cpu().numpy()
        probs = probs.reshape(8, 8, len(CLASSIFIER_CLASSES))
        return probs


def main():
    detector = MobileDetector(checkpoint_path='ChessClassifier/cq4zp3xk/checkpoints/epoch=3-step=20040.ckpt',
                              device='cuda')
    dataset = 'google'

    label_paths = list(glob(os.path.join(DATA_DIR, dataset, 'labels', '*')))
    for label_path in tqdm(label_paths, desc=dataset):
        with open(label_path, 'r') as f:
            label = json.load(f)
        if 'keypoints' not in label:
            continue

        image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
        image = Image.open(image_path)

        keypoints = np.array(list(label['keypoints'].values()))
        keypoints[:, 0] *= image.width
        keypoints[:, 1] *= image.height

        probs = detector.run(image, keypoints)
        print(probs)
        exit(1)



if __name__ == '__main__':
    main()