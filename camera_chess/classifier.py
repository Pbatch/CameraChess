from collections import namedtuple

import cv2
import numpy as np
import onnxruntime
from PIL import Image
from scipy.spatial import KDTree
from shapely.geometry import Polygon

from camera_chess.constants import CLASSES, BOARD_SIZE, SQUARE_SIZE
from camera_chess.utils import get_square

classification = namedtuple("Classification", "bbox conf piece center square")


class Classifier:
    def __init__(self, model_path, keypoints, conf_thres=0.2):
        self.model_path = model_path
        self.keypoints = keypoints
        self.conf_thres = conf_thres

        self.sess = onnxruntime.InferenceSession(self.model_path)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.onnx_height, self.onnx_width = self.sess.get_inputs()[0].shape[-2:]
        self.square_centers = self._get_square_centers()
        self.kd_tree = KDTree(self.square_centers)
        self.roi = Polygon(self.keypoints)

    @staticmethod
    def _zero_king_scores(pred):
        for piece in ['white-king', 'black-king']:
            idx = 4 + CLASSES.index(piece)
            best_idx = pred[4:, idx].argmax()
            best_score = pred[4 + best_idx, idx]
            pred[:, idx] = 0
            pred[4 + best_idx, idx] = best_score
        return pred

    @staticmethod
    def _zero_pawn_scores(pred, squares):
        pawn_idx = [4 + CLASSES.index(p) for p in ['white-pawn', 'black-pawn']]
        for p, square in zip(pred, squares):
            if square[1] in {'1', '8'}:
                p[pawn_idx] = 0
        return pred

    def _get_square_centers(self):
        target = np.array([[BOARD_SIZE, BOARD_SIZE],
                           [0, BOARD_SIZE],
                           [0, 0],
                           [BOARD_SIZE, 0]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(self.keypoints, target)
        inv_matrix = np.linalg.inv(matrix)
        grid = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        square_centers = cv2.perspectiveTransform(np.expand_dims(grid, axis=0), inv_matrix)[0]
        return square_centers

    def _preprocess_image(self, image):
        image = image.resize((self.onnx_width, self.onnx_height), Image.BICUBIC)
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        image = image / 255
        return image

    def _run_od(self, np_image, width, height):
        pred = self.sess.run([self.output_name], {self.input_name: np_image})[0][0]
        pred = pred.transpose((1, 0))
        pred[:, 0] -= pred[:, 2] / 2
        pred[:, 1] -= pred[:, 3] / 2
        pred[:, 2] += pred[:, 0]
        pred[:, 3] += pred[:, 1]
        pred[:, [0, 2]] *= width / self.onnx_width
        pred[:, [1, 3]] *= height / self.onnx_height
        return pred

    def _filter_by_confidence(self, pred):
        scores = pred[:, 4:]
        best_scores_idx = np.argmax(scores, axis=1).reshape(-1, 1)
        best_scores = np.take_along_axis(scores, best_scores_idx, axis=1)
        mask = np.ravel(best_scores > self.conf_thres)
        pred = pred[mask]
        return pred

    def _filter_by_roi(self, pred):
        pred = np.array([p
                         for p in pred
                         if Polygon([(p[0], p[1]), (p[0], p[3]), (p[2], p[3]), (p[2], p[1])]).intersects(self.roi)])
        return pred

    def _filter_by_square(self, pred):
        piece_centers = np.vstack([(pred[:, 0] + pred[:, 2]) / 2,
                                   pred[:, 3] - ((pred[:, 2] - pred[:, 0]) / 4)]).T
        distances, idxs = self.kd_tree.query(piece_centers)
        matches = {}
        for i in range(len(piece_centers)):
            if idxs[i] in matches and distances[i] > matches[idxs[i]][0]:
                continue
            matches[idxs[i]] = [distances[i], i]
        matches = {v[1]: k for k, v in matches.items()}

        mask = np.array(list(matches.keys()))
        pred = pred[mask]
        piece_centers = piece_centers[mask]
        squares = [get_square(i) for i in matches.values()]

        return pred, piece_centers, squares

    def _post_process_pred(self, pred):
        pred = self._filter_by_confidence(pred)
        pred = self._filter_by_roi(pred)
        pred, piece_centers, squares = self._filter_by_square(pred)
        pred = self._zero_king_scores(pred)
        pred = self._zero_pawn_scores(pred, squares)

        clean_pred = []
        for p, center, square in zip(pred, piece_centers, squares):
            bbox = np.round(p[:4]).astype(int)
            best_idx = np.argmax(p[4:])
            piece = CLASSES[best_idx]
            conf = p[4 + best_idx]
            clean_pred.append(classification(bbox, conf, piece, center, square))

        return clean_pred

    def run(self, image):
        np_image = self._preprocess_image(image)
        pred = self._run_od(np_image, image.width, image.height)
        pred = self._post_process_pred(pred)
        return pred
