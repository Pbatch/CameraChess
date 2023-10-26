import os
from glob import glob

import cv2 as cv2
import numpy as np
import torch
import torchvision
from icecream import ic
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from tqdm import tqdm

from camera_chess.constants import YOLO_DIR, MODEL_DIR
from camera_chess.export.wrapped_model import load_model


def find_offset(warped_xcorners):
    best_score = 0
    best_offset = None
    grid = np.concatenate(np.meshgrid(np.arange(-6, 7), np.arange(-6, 7))).reshape(2, -1).T.astype(np.float32)
    dist = cdist(warped_xcorners, grid)

    for i in range(7):
        for j in range(7):
            shift = 7 * i + 7 * j
            print(49 + shift, dist.shape)
            row_idx, col_idx = linear_sum_assignment(dist[:, shift:49 + shift])
            score = 1 / dist[row_idx, col_idx + shift].sum()
            if score > best_score:
                best_score = score
                best_offset = [i - 6, j - 6]
    exit(1)
    return best_offset, best_score


def apply_transform(src, transform):
    return cv2.perspectiveTransform(np.expand_dims(src, 0).astype(np.float32),
                                    transform.astype(np.float32))[0].astype(np.float32)


def refine_transform(xcorners, transform):
    warped_xcorners = apply_transform(xcorners, transform)
    refined_transform, _ = cv2.findHomography(xcorners, warped_xcorners, cv2.RANSAC)
    if refined_transform is None:
        refined_transform = transform
    return refined_transform


def score_quad(quad, xcorners):
    ideal_quad = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), ideal_quad)

    M_refined = refine_transform(xcorners, M)
    warped_xcorners = apply_transform(xcorners, M_refined)

    offset, score = find_offset(warped_xcorners)

    return score, M, offset


def get_all_quads(xcorners):
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


def brutesac_chessboard(xcorners):
    quads = get_all_quads(xcorners)

    best_score = 0
    best_M = None
    best_quad = None
    best_offset = None
    for quad in xcorners[quads]:
        score, M, offset = score_quad(quad, xcorners)

        warped_xcorners = apply_transform(xcorners, M).round() - offset

        outliers = np.any((warped_xcorners < 0) | (warped_xcorners > 7), axis=1)
        refined_M, _ = cv2.findHomography(xcorners[~outliers], warped_xcorners[~outliers], cv2.LMEDS)
        if refined_M is None:
            refined_M = M

        warped_xcorners = apply_transform(xcorners, refined_M)
        offset, score = find_offset(warped_xcorners)

        if score > best_score:
            best_score = score
            best_M = refined_M
            best_quad = quad
            best_offset = offset

    return best_M, best_quad, best_score, best_offset


def get_xcorners(image, model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    iou_threshold = 0.1
    conf_threshold = 0.3
    images = torch.tensor(np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0), device='cuda:0')
    pred = model(images)[0]
    boxes = pred[:, :4]
    scores = pred[:, 4]

    mask = (scores > conf_threshold).ravel()
    boxes = boxes[mask]
    scores = scores[mask]

    keep = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
    boxes = boxes[keep].detach().cpu().numpy().astype(int)

    xcorners = []
    for l, t, r, b in boxes:
        region = dst[t:b, l:r]
        dx, dy = np.unravel_index(region.argmax(), region.shape)
        xcorner = [l + dx, t + dy]
        xcorners.append(xcorner)
    xcorners = np.array(xcorners, dtype=np.float32)

    return xcorners


def elucidation(xcorners):
    M, quad, score, offset = brutesac_chessboard(xcorners)
    inv_M = np.linalg.inv(M)

    corners = np.array([[-1, -1], [-1,  7], [7,  7], [7, -1], [-1, -1]]) + offset
    unwarped_corners = apply_transform(corners, inv_M)

    grid = np.concatenate(np.meshgrid(np.arange(7), np.arange(7))).reshape(2, -1).T + offset
    unwarped_grid = apply_transform(grid, inv_M)

    return unwarped_corners, unwarped_grid, quad, score, offset


def main():
    model = load_model(model_path=os.path.join(MODEL_DIR, "480L_keypoints.pt"), device='cuda:0')
    label_paths = list(glob(os.path.join(YOLO_DIR, 'val', 'labels', '*')))
    # label_paths = [os.path.join(YOLO_DIR, 'val', 'labels', 'google_5.jpg')]

    os.makedirs('debug', exist_ok=True)
    for label_path in tqdm(label_paths):
        image_path = label_path.replace('labels', 'images').replace('.txt', '.jpg')
        image = cv2.imread(image_path)

        # Use gt
        # with open(label_path, 'r') as f:
        #     boxes = np.array([[float(i) for i in line.strip().split()[1:]] for line in f.readlines()])
        # TODO: Fix

        xcorners = get_xcorners(image, model)
        if len(xcorners) < 4:
            print(f'Not enough xcorners for "{image_path}". Skipping...')
            continue

        unwarped_corners, unwarped_grid, quad, score, offset = elucidation(xcorners)

        for i in range(4):
            cv2.line(image, unwarped_corners[i].astype(int), unwarped_corners[i + 1].astype(int), color=(0, 0, 255),
                     thickness=5)
        for p in xcorners:
            image = cv2.circle(image, p.astype(int), radius=1, color=(255, 0, 0), thickness=3)
        for p in unwarped_grid:
            image = cv2.circle(image, p.astype(int), radius=1, color=(0, 0, 255), thickness=3)
        for p in quad:
            image = cv2.circle(image, p.astype(int), radius=1, color=(0, 255, 0), thickness=3)

        save_path = os.path.join('debug', os.path.basename(image_path))
        ic(save_path, score)
        cv2.imwrite(save_path, image)
        exit(1)


if __name__ == '__main__':
    main()
