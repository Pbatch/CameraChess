import json
import os

import chess
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from camera_chess.board_detector import BoardDetector
from camera_chess.constants import BOARD_SIZE, SQUARE_SIZE, CLASSES, ABBR_MAP, DATA_DIR, CORNERS
from camera_chess.detector import Detector
from camera_chess.utils import load_video_config, update_state, draw_lines, draw_text
from camera_chess.video import Video


def perspective_transform(src, matrix):
    if src.ndim == 2:
        src = np.expand_dims(src, axis=0)
    homo_src = src.transpose(1, 0, 2)
    homo_src = np.concatenate([homo_src, np.ones((len(homo_src), 1, 1))], axis=2)
    warped_src = homo_src @ matrix.T
    warped_src /= warped_src[..., 2:]
    warped_src = warped_src[..., :2]
    warped_src = warped_src.transpose(1, 0, 2)
    return warped_src


def get_perspective_transform(target, keypoints):
    A = np.zeros((8, 8), dtype=np.float32)
    B = np.zeros((8, 1), dtype=np.float32)

    for i in range(4):
        x, y = keypoints[i]
        u, v = target[i]
        A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        B[i * 2] = u
        B[i * 2 + 1] = v

    matrix = np.linalg.solve(A, B)
    matrix = np.append(matrix, 1.0)
    matrix = matrix.reshape((3, 3))
    return matrix


def get_box_centers(preds):
    cx = (preds[:, 0] + preds[:, 2]) / 2
    cy = preds[:, 3] - ((preds[:, 2] - preds[:, 0]) / 3)
    box_centers = np.vstack((cx, cy)).T
    return box_centers


def get_centers_and_boundary(keypoints):
    target = np.array([[BOARD_SIZE, BOARD_SIZE],
                       [0, BOARD_SIZE],
                       [0, 0],
                       [BOARD_SIZE, 0]], dtype=np.float32)
    matrix = get_perspective_transform(target, keypoints)
    inv_matrix = np.linalg.inv(matrix)

    x = np.linspace(0.5, 7.5, num=8)
    y = np.linspace(7.5, 0.5, num=8)
    warped_centers = np.concatenate(np.meshgrid(x, y)).reshape(2, -1).T * SQUARE_SIZE
    centers = perspective_transform(warped_centers, inv_matrix)[0]

    warped_boundary = np.array([[-0.5, -0.5], [-0.5, 8.5], [8.5, 8.5], [8.5, -0.5]]) * SQUARE_SIZE
    boundary = perspective_transform(warped_boundary, inv_matrix)[0]

    return centers, boundary


def get_squares(centers, box_centers, oob):
    squares = -np.ones(len(box_centers), dtype=np.int32)
    dist = np.sum(np.square(np.expand_dims(box_centers[~oob], 1) - np.expand_dims(centers, 0)), axis=2)
    squares[~oob] = np.argmin(dist, axis=1)
    return squares


def get_oob(boundary, box_centers):
    oob = np.zeros(len(box_centers), dtype=bool)
    for i in range(4):
        a = boundary[i - 1][0] - boundary[i][0]
        b = boundary[i - 1][1] - boundary[i][1]
        c = box_centers[:, 0] - boundary[i][0]
        d = box_centers[:, 1] - boundary[i][1]
        cross_product = (a * d) - (b * c)
        oob[cross_product < 0] = True
    return oob


def process_preds(preds, conf, boundary, centers, frame=0, sequence=None):
    box_centers = get_box_centers(preds)
    oob = get_oob(boundary, box_centers)
    squares = get_squares(centers, box_centers, oob)

    if sequence is not None:
        for square, pred in zip(squares[~oob], preds[~oob]):
            if conf is None:
                cls_idx = int(pred[5])
                sequence[frame][square][cls_idx] = max(sequence[frame][square][cls_idx], pred[4])
            else:
                sequence[frame][square] = np.maximum(sequence[frame][square], pred[4:])
    squares[oob] = -1

    if conf is not None:
        sorted_idx = (-conf[~oob]).argsort()
        unique_idx = np.unique(squares[~oob][sorted_idx], return_index=True)[1]
        non_oob_keep = np.where(~oob)[0][sorted_idx][unique_idx]

        nms_idx = torchvision.ops.nms(boxes=torch.tensor(preds[oob, :4]),
                                      scores=torch.tensor(conf[oob]),
                                      iou_threshold=0.5).detach().cpu().numpy()
        oob_keep = np.where(oob)[0][nms_idx]

        keep = list(set(non_oob_keep) | set(oob_keep))
        idx = frame * np.ones((len(keep), 1), dtype=np.int32)
        squares = np.expand_dims(squares[keep], axis=1)
        frame_boxes = preds[keep, :4]
        max_conf = np.expand_dims(conf[keep], axis=1)
        cls = np.expand_dims(np.argmax(preds[keep, 4:], axis=1), axis=1)
        frame_info = np.concatenate([idx, squares, frame_boxes, cls, max_conf], axis=1).tolist()
    else:
        idx = frame * np.ones((len(preds), 1), dtype=np.int32)
        squares = np.expand_dims(squares, axis=1)
        frame_boxes = preds[:, :4]
        max_conf = np.expand_dims(preds[:, 4], axis=1)
        cls = np.expand_dims(preds[:, 5], axis=1)
        frame_info = np.concatenate([idx, squares, frame_boxes, cls, max_conf], axis=1).tolist()

    return frame_info


class SequenceGenerator:
    PLOT_SIZE = 64

    def __init__(self, dataset, model_basename):
        self.dataset = dataset
        self.model_basename = model_basename

        self.video_config = load_video_config(self.dataset)
        self.video = Video(self.video_config, target_fps=8)

        if self.video_config.keypoints is not None:
            self.centers, self.boundary = get_centers_and_boundary(self.video_config.keypoints)

        cmap = plt.get_cmap('Blues')
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        self.scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

        save_dir = os.path.join(DATA_DIR, dataset, model_basename.split('.')[0])
        os.makedirs(save_dir, exist_ok=True)
        self.sequence_path = os.path.join(save_dir, 'sequence.npy')
        self.boxes_path = os.path.join(save_dir, 'boxes.npy')
        self.video_path = os.path.join(save_dir, 'debug.mp4')
        self.sequence_video_path = os.path.join(save_dir, 'sequence_video.avi')

        self.v10 = "v10" in self.model_basename

    def _plot_state(self, state, from_square, to_square, thr=0.7):
        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        image = Image.new('RGB', size)
        d = ImageDraw.Draw(image)
        max_idx = np.argmax(state, axis=1)
        for square in range(64):
            cls = max_idx[square]
            score = state[square][cls]

            x = square % 8
            y = 7 - (square // 8)
            bbox = [self.PLOT_SIZE * i for i in [x, y, (x + 1), (y + 1)]]
            fill = colors.rgb2hex(self.scalar_map.to_rgba((score - thr) / (1 - thr)))

            if square in {from_square, to_square}:
                outline = "yellow"
                width = 5
            else:
                outline = "black"
                width = 1
            d.rectangle(bbox, fill=fill, outline=outline, width=width)

            if score > thr:
                d.text([self.PLOT_SIZE * (x + 0.45), self.PLOT_SIZE * (y + 0.45)],
                       ABBR_MAP[CLASSES[cls]],
                       fill='black')
        return image

    def create_sequence(self, debug=False, force=False, infer_keypoints=False):
        if os.path.isfile(self.sequence_path) and os.path.isfile(self.boxes_path) and not force:
            sequence = np.load(self.sequence_path)
            boxes = np.load(self.boxes_path)
            return sequence, boxes

        if debug:
            debug_video = cv2.VideoWriter(self.sequence_video_path, 0, 10, (self.video.width, self.video.height))

        detector = Detector(model_basename=self.model_basename)
        board_detector = BoardDetector()
        sequence = np.zeros((len(self.video), 64, len(CLASSES)))
        boxes = []

        if not infer_keypoints and self.video_config.keypoints is None:
            print(f'No keypoints for {self.dataset}. Switching to infer mode.')
            infer_keypoints = True

        if infer_keypoints:
            keypoints = None
        else:
            keypoints = {k: v for k, v in zip(CORNERS, self.video_config.keypoints)}

        for i, (image, frame) in tqdm(enumerate(self.video),
                                      desc=f'Creating sequence for {self.dataset}',
                                      total=len(self.video)):
            preds = detector.run(image, keypoints)

            if self.v10:
                preds = preds[preds[:, 4] > 0.1]
                conf = None
            else:
                conf = np.max(preds[:, 4:], axis=1)
                conf_mask = conf > 0.1
                preds = preds[conf_mask]
                conf = conf[conf_mask]

            if infer_keypoints:
                corners, _ = board_detector.find_corners(image)
                if corners is None:
                    tqdm.write(f'Bad frame {frame}')
                    continue
                if keypoints is None:
                    keypoints = board_detector.match_corners_using_preds(corners, preds)
                else:
                    keypoints = board_detector.match_corners_using_keypoints(corners, keypoints)
                self.centers, self.boundary = get_centers_and_boundary([keypoints[k] for k in CORNERS])

            frame_info = process_preds(preds, conf, self.boundary, self.centers,
                                       frame=i, sequence=sequence)
            boxes.extend(frame_info)

            if debug:
                pil_image = Image.fromarray(image)
                d = ImageDraw.Draw(pil_image)
                draw_lines(d, list(keypoints.values()), colour='red')
                for text, (x, y) in keypoints.items():
                    bbox = [x - 5, y - 5, x + 5, y + 5]
                    draw_text(d, bbox, text)

                for info in frame_info:
                    bbox = info[2:6]
                    cls = int(info[6])
                    conf = info[7]
                    text = f'{CLASSES[cls]}:{conf:.2f}'
                    draw_text(d, bbox, text=text)

                debug_video.write(np.array(pil_image)[..., ::-1])

        sequence = np.asarray(sequence)
        np.save(self.sequence_path, sequence)

        boxes = np.asarray(boxes)
        boxes = boxes[(-boxes[:, -1]).argsort()]
        np.save(self.boxes_path, boxes)

        cv2.destroyAllWindows()
        if debug:
            debug_video.release()

        return sequence, boxes

    def create_video(self, logs_path=None):
        sequence = np.load(self.sequence_path)

        if logs_path is not None:
            with open(logs_path, 'rb') as f:
                logs = json.load(f)
        else:
            logs = {}

        size = (8 * self.PLOT_SIZE + 1, 8 * self.PLOT_SIZE + 1)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(self.video_path, fourcc, self.video.target_fps, size)

        state = np.zeros((64, len(CLASSES)), dtype=np.float32)
        from_square = None
        to_square = None
        board = chess.Board(self.video_config.fen)
        for i in tqdm(range(len(sequence)), desc='Writing debug video'):
            update_state(state, sequence[i])

            if str(i) in logs:
                d = logs[str(i)]
                uci_move = board.parse_san(d['moves'].split()[0])
                board.push(uci_move)
                from_square = uci_move.from_square
                to_square = uci_move.to_square

            image = self._plot_state(state, from_square, to_square)
            data = np.array(image)
            writer.write(data[..., ::-1])

        writer.release()
