import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from icecream import ic
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from tqdm import tqdm

from camera_chess.constants import MODEL_DIR, YOLO_DIR, CORNERS
from camera_chess.export.wrapped_model import load_model


class BoardDetector:
    GRID = np.concatenate(np.meshgrid(np.arange(7), np.arange(7))).reshape(2, -1).T.astype(np.float32)

    def __init__(self, model_basename="480L_xcorner.pt", device='cuda:0', iou_thresh=0.1, conf_thresh=0.3):
        self.model_basename = model_basename
        self.device = device
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

        self.model = load_model(model_path=os.path.join(MODEL_DIR, self.model_basename),
                                device=self.device)

    @staticmethod
    def _apply_transform(src, transform):
        return cv2.perspectiveTransform(np.expand_dims(src, 0).astype(np.float32),
                                        transform.astype(np.float32))[0].astype(np.float32)

    @staticmethod
    def _run_delaunay(xcorners):
        tri = Delaunay(xcorners)
        quads = []
        pairings = set()
        for i, neighbors in enumerate(tri.neighbors):
            for k, nk in enumerate(neighbors):
                if nk == -1:
                    continue
                pair = (i, nk)
                reverse_pair = (nk, i)
                if reverse_pair not in pairings:
                    pairings.add(pair)
                    b = tri.simplices[i]
                    d = tri.simplices[nk]
                    nk_vtx = (set(d) - set(b)).pop()
                    insert_mapping = [2, 3, 1]
                    b = np.insert(b, insert_mapping[k], nk_vtx)
                    quads.append(b)

        return np.array(quads)

    def _calculate_offset_score(self, warped_xcorners, shift):
        dist = cdist(warped_xcorners, self.GRID + shift)
        row_idx, col_idx = linear_sum_assignment(dist)
        score = 1 / (1 + dist[row_idx, col_idx].sum())
        return score

    def _find_offset(self, warped_xcorners):
        best_offset = [0, 0]
        for i in range(2):
            low = -7
            high = 1
            scores = {}
            while high - low > 1:
                mid = (high + low) >> 1
                if mid not in scores:
                    shift = [0, 0]
                    shift[i] = mid
                    scores[mid] = self._calculate_offset_score(warped_xcorners, shift)
                if mid + 1 not in scores:
                    shift = [0, 0]
                    shift[i] = mid + 1
                    scores[mid + 1] = self._calculate_offset_score(warped_xcorners, shift)
                if scores[mid] > scores[mid + 1]:
                    high = mid
                else:
                    low = mid
            best_offset[i] = low + 1

        return best_offset

    def _score_quad(self, quad, xcorners):
        # Initial transform
        ideal_quad = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), ideal_quad)

        # First attempt
        warped_xcorners = self._apply_transform(xcorners, M).round()
        offset = self._find_offset(warped_xcorners)

        # Second attempt - remove outliers
        warped_xcorners = self._apply_transform(xcorners, M).round() - offset
        outliers = np.any((warped_xcorners < 0) | (warped_xcorners > 7), axis=1)
        if len(xcorners[~outliers]) >= 4:
            refined_M, _ = cv2.findHomography(xcorners[~outliers], warped_xcorners[~outliers], cv2.LMEDS)
            if refined_M is not None:
                M = refined_M

        # Score final matrix
        # Do not round the warped xcorners
        warped_xcorners = self._apply_transform(xcorners, M)
        offset = self._find_offset(warped_xcorners)
        score = self._calculate_offset_score(warped_xcorners, offset)

        return score, M, offset

    def _find_transform(self, xcorners):
        quads = self._run_delaunay(xcorners)

        best_score = 0
        best_M = None
        best_quad = None
        best_offset = None
        for quad in xcorners[quads]:
            score, M, offset = self._score_quad(quad, xcorners)
            if score > best_score:
                best_score = score
                best_M = M
                best_quad = quad
                best_offset = offset

        return best_M, best_quad, best_score, best_offset

    def _find_xcorners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        images = torch.tensor(np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0),
                              device=self.device)
        pred = self.model(images)[0]
        boxes = pred[:, :4]
        scores = pred[:, 4]

        mask = (scores > self.conf_thresh).ravel()
        boxes = boxes[mask]
        scores = scores[mask]

        keep = torchvision.ops.nms(boxes, scores, iou_threshold=self.iou_thresh)
        boxes = boxes[keep].detach().cpu().numpy()
        height, width = image.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=boxes[:, 2])
        boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=boxes[:, 3])
        boxes[:, 2] = np.clip(boxes[:, 2], a_min=boxes[:, 0], a_max=width)
        boxes[:, 3] = np.clip(boxes[:, 3], a_min=boxes[:, 1], a_max=height)
        boxes = boxes.astype(int)

        xcorners = []
        for left, top, right, bottom in boxes:
            if right - left == 0 or bottom - top == 0:
                continue
            region = dst[top:bottom, left:right]
            dx, dy = np.unravel_index(region.argmax(), region.shape)
            xcorner = [left + dx, top + dy]
            xcorners.append(xcorner)
        xcorners = np.array(xcorners, dtype=np.float32)

        return xcorners

    def _create_corners(self, M, offset):
        inv_M = np.linalg.inv(M)

        warped_corners = np.array([[-1, -1], [-1, 7], [7, 7], [7, -1]]) + offset
        corners = self._apply_transform(warped_corners, inv_M)
        return corners

    def find_corners(self, image):
        xcorners = self._find_xcorners(image)
        if len(xcorners) < 4:
            corners = None
        else:
            M, quad, score, offset = self._find_transform(xcorners)
            corners = self._create_corners(M, offset)
        return corners

    def match_corners_using_preds(self, corners, preds):
        cx = (preds[:, 0] + preds[:, 2]) / 2
        cy = preds[:, 3] - ((preds[:, 2] - preds[:, 0]) / 3)
        box_centers = np.vstack((cx, cy)).T

        dist = cdist(corners, box_centers)
        black_conf = np.max(preds[:, 4:10], axis=1)
        white_conf = np.max(preds[:, 10:], axis=1)

        white_scores = np.dot(1/dist, white_conf)
        black_scores = np.dot(1/dist, black_conf)
        scores = white_scores - black_scores

        best_i = None
        best_score = -float('inf')
        for i in range(4):
            rolled_scores = np.roll(scores, i)
            score = rolled_scores[0] + rolled_scores[1] - rolled_scores[2] - rolled_scores[3]
            if score > best_score:
                best_score = score
                best_i = i

        keypoints = {k: corners[v].tolist() for k, v in zip(CORNERS, np.roll([0, 1, 2, 3], best_i))}

        return keypoints

    def match_corners_using_keypoints(self, corners, keypoints):
        dist = cdist(corners, list(keypoints.values()))
        row_idx, col_idx = linear_sum_assignment(dist)

        keys = list(keypoints.keys())
        keypoints = {keys[j]: corners[i].tolist() for i, j in zip(row_idx, col_idx)}

        return keypoints


def main():
    label_paths = list(glob(os.path.join(YOLO_DIR, 'val', 'labels', '*')))
    detector = BoardDetector()

    os.makedirs('debug', exist_ok=True)
    for label_path in tqdm(label_paths):
        image_path = label_path.replace('labels', 'images').replace('.txt', '.jpg')
        image = cv2.imread(image_path)
        corners = detector.find_corners(image)

        if corners is not None:
            for i in range(4):
                cv2.line(image, corners[i - 1].astype(int), corners[i].astype(int), color=(0, 0, 255),
                         thickness=5)

        save_path = os.path.join('debug', os.path.basename(image_path))
        cv2.imwrite(save_path, image)


if __name__ == '__main__':
    main()
