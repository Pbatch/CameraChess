import json

import pandas as pd
import torch
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from camera_chess.calibration.calibrator import Calibrator, platt_scaling_logits
from camera_chess.constants import DATA_DIR, CLASSES, ROOT_DIR
from camera_chess.detector import Detector

CALIBRATION_DIR = os.path.join(ROOT_DIR, 'camera_chess', 'calibration')


def generate_preds():
    os.makedirs("preds", exist_ok=True)
    detector = Detector("480S_pieces_480x288.onnx")

    image_paths = sorted(glob(os.path.join(DATA_DIR, 'pieces', 'train', 'images', '*.jpg')))
    for path in tqdm(image_paths):
        image = np.array(Image.open(path))
        height, width = image.shape[:2]
        keypoints = {"h1": np.array([0, 0]),
                     "a1": np.array([width, 0]),
                     "a8": np.array([width, height]),
                     "h8": np.array([0, height])}
        res = detector.run(image, keypoints)
        res[:, [0, 2]] /= width
        res[:, [1, 3]] /= height

        save_path = os.path.join("preds", os.path.basename(path).replace('.jpg', '.npy'))
        np.save(save_path, res)


def load_gt(gt_path):
    with open(gt_path) as f:
        lines = [line.strip() for line in f.readlines()]

    gt = []
    for line in lines:
        cls, xc, yc, w, h = [float(i) for i in line.split()]
        left = xc - w / 2
        right = xc + w / 2
        top = yc - h / 2
        bottom = yc + h / 2
        gt.append([left, top, right, bottom, cls])

    if len(gt) == 0:
        gt = torch.empty((0, 5), device='cuda')
    else:
        gt = torch.tensor(np.array(gt), device='cuda')

    return gt


def load_pred(pred_path, conf_thres=0.1):
    res = []

    pred = np.load(pred_path)
    for p in pred:
        bbox = p[:4]
        probs = p[4:]
        idx = np.argwhere(probs > conf_thres)
        if len(idx) == 0:
            continue

        for i in idx[0]:
            res.append([*bbox, i, probs[i]])

    if len(res) == 0:
        res = torch.empty((0, 6), device='cuda')
    else:
        res = torch.tensor(np.array(res), device='cuda')

    return res


def calculate_iou(bboxes_1, bboxes_2):
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0]) * (bboxes_1[:, 3] - bboxes_1[:, 1])
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1])
    mini = torch.min(bboxes_1[:, None, 2:], bboxes_2[:, 2:])
    maxi = torch.max(bboxes_1[:, None, :2], bboxes_2[:, :2])
    intersection = (mini - maxi).clamp(0).prod(2)
    iou = intersection / (area_1[:, None] + area_2 - intersection)
    return iou


def create_match_df(gt_iou_thres=0.5):
    gt_paths = sorted(glob(os.path.join(DATA_DIR, 'pieces', 'train', 'labels', '*.txt')))
    match_data = []
    for gt_path in tqdm(gt_paths):
        pred_path = os.path.join("preds", os.path.basename(gt_path).replace('.txt', '.npy'))
        gts = load_gt(gt_path)
        preds = load_pred(pred_path)
        iou = calculate_iou(preds[:, :4], gts[:, :4])
        mask = iou > gt_iou_thres
        matches = torch.argwhere(mask)
        pred_idx_to_gt_idx = dict(zip(*matches.T.detach().cpu().numpy()))
        for pred_idx, pred in enumerate(preds):
            if pred_idx in pred_idx_to_gt_idx:
                gt_idx = pred_idx_to_gt_idx[pred_idx]
                gt_cls = CLASSES[int(gts[gt_idx, 4])]
            else:
                gt_cls = -1
            pred_conf = pred[5].item()
            pred_cls = CLASSES[int(pred[4])]
            gt_conf = float(gt_cls == pred_cls)
            match_data.append({
                f'gt_{pred_cls}': gt_conf,
                f'pred_{pred_cls}': pred_conf
            })

    match_df = pd.DataFrame.from_records(match_data)
    match_df.fillna(-1.0, inplace=True)

    save_path = os.path.join(CALIBRATION_DIR, 'match_df.csv')
    match_df.to_csv(save_path, index=False)


def run_calibration(n_bins=20, plots_dir=None, map_function=None):
    if map_function is None:
        map_function = platt_scaling_logits

    if plots_dir is None:
        plots_dir = os.path.join(CALIBRATION_DIR, 'plots')

    match_df = pd.read_csv(os.path.join(CALIBRATION_DIR, 'match_df.csv'))
    calibrator = Calibrator(match_df=match_df, n_bins=n_bins)
    calibrator.generate_plots(plots_dir=plots_dir, suffix="")

    results = calibrator.minimize_log_loss(map_function=map_function,
                                           x0=np.array([1, 1]))
    for attr, d in results.items():
        params = results[attr]["params"]
        res = map_function(match_df.loc[:, f'pred_{attr}'].values,
                           bias=params['bias'],
                           conf_weight=params['conf_weight'])
        match_df.loc[:, f'pred_{attr}'] = res.detach().cpu().numpy()

    calibrator = Calibrator(match_df=match_df, n_bins=n_bins)
    attr_to_stats = calibrator.generate_plots(plots_dir=plots_dir, suffix="_calibrated")
    for attr, stats in attr_to_stats.items():
        results[attr]['dece'] = calibrator.compute_metric(stats)

    save_path = os.path.join(CALIBRATION_DIR, 'results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    # generate_preds()
    create_match_df()
    run_calibration()


if __name__ == '__main__':
    main()
