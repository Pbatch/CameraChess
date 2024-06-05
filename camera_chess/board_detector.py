import json
import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from tqdm import tqdm

from camera_chess.constants import CORNERS, DATA_DIR
from camera_chess.detector import Detector
from camera_chess.utils import draw_lines, draw_points


class BoardDetector:
    GRID = np.concatenate(np.meshgrid(np.arange(7), np.arange(7))).reshape(2, -1).T.astype(np.float32)

    def __init__(self, model_basename="480L_xcorners_480x288.onnx", device='cuda:0', iou_thresh=0.1, conf_thresh=0.3):
        self.model_basename = model_basename
        self.device = device
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

        self.model = Detector(model_basename=self.model_basename,
                              device=self.device)

    @staticmethod
    def _apply_transform(src, transform):
        src_3d = np.concatenate([src, np.ones([len(src), 1])], axis=1)
        warped_src = src_3d @ transform.T
        warped_src /= warped_src[:, 2][:, None] + 1e-8
        warped_src = warped_src[:, :2]
        return warped_src

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
        dist = cdist(self.GRID + shift, warped_xcorners)
        cost = np.sum(np.min(dist, axis=1))
        score = 1 / (1 + cost)
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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        keypoints = {"h1": np.array([0, 0]),
                     "a1": np.array([width, 0]),
                     "a8": np.array([width, height]),
                     "h8": np.array([0, height])}
        pred = torch.tensor(self.model.run(image, keypoints))
        boxes = pred[:, :4]
        scores = pred[:, 4]

        mask = (scores > self.conf_thresh).ravel()
        boxes = boxes[mask]
        scores = scores[mask]

        keep = torchvision.ops.nms(boxes, scores, iou_threshold=self.iou_thresh)
        boxes = boxes[keep].detach().cpu().numpy()
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
        if len(xcorners) < 5:
            return None, None

        M, quad, score, offset = self._find_transform(xcorners)
        corners = self._create_corners(M, offset)
        return corners, xcorners

    @staticmethod
    def match_corners_using_preds(corners, preds):
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

    @staticmethod
    def match_corners_using_keypoints(corners, keypoints):
        dist = cdist(corners, list(keypoints.values()))
        row_idx, col_idx = linear_sum_assignment(dist)

        keys = list(keypoints.keys())
        keypoints = {keys[j]: corners[i].tolist() for i, j in zip(row_idx, col_idx)}

        return keypoints


def main():
    label_paths = list(glob(os.path.join(DATA_DIR, 'google', 'labels', '*')))
    detector = BoardDetector()

    os.makedirs('debug', exist_ok=True)
    total_score = 0
    for label_path in tqdm(label_paths):
        image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
        image = cv2.imread(image_path)

        pred, xcorners = detector.find_corners(image)
        if pred is None:
            continue

        with open(label_path, 'rb') as f:
            label = json.load(f)
        gt = np.array(list(label['keypoints'].values()))
        height, width = image.shape[:2]
        gt[:, 0] *= width
        gt[:, 1] *= height

        dist = cdist(gt, pred)
        row_idx, col_idx = linear_sum_assignment(dist)
        score = width * height / (1000 * np.sum(dist[row_idx, col_idx]))
        total_score += score
        basename = os.path.basename(image_path)
        tqdm.write(f"{basename}: {score:.3f}")

        pil_image = Image.fromarray(image[..., ::-1])
        d = ImageDraw.Draw(pil_image)
        draw_lines(d, pred, colour='red')
        draw_lines(d, gt, colour='blue')
        draw_points(d, xcorners, colour="green")
        save_path = os.path.join('debug', basename)
        pil_image.save(save_path)

    tqdm.write(f"Total: {total_score:.3f}")


if __name__ == '__main__':
    main()
