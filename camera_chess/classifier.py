import os
from collections import namedtuple

import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type
from scipy.spatial import KDTree

from camera_chess.constants import CLASSES, SQUARE_SIZE
from camera_chess.utils import warp, get_square

classification = namedtuple("Classification", "bbox conf piece center square")


class Classifier:
    def __init__(self, model_path, keypoints, conf_thres=0.2):
        self.model_path = model_path
        self.keypoints = keypoints
        self.conf_thres = conf_thres

        self.model = Core().read_model(model=self.model_path)
        self.vino_height = 480
        self.vino_width = 480

        self.kd_tree = None
        self.compiled_model = None
        self.output_layer_ir = None
        self.input_width = 0
        self.input_height = 0

        self.set_kd_tree()

    def set_compiled_model(self):
        core = Core()
        ppp = PrePostProcessor(self.model)
        model_input = ppp.input('images')
        model_input.tensor()\
            .set_layout(Layout('NHWC'))\
            .set_shape([1, self.input_height, self.input_width, 3])\
            .set_element_type(Type.u8)
        model_input.preprocess()\
            .convert_element_type(Type.f16)\
            .scale([255])\
            .resize(ResizeAlgorithm.RESIZE_LINEAR, self.vino_width, self.vino_height)
        model_input.model().set_layout(Layout('NCHW'))
        ppp_model = ppp.build()
        self.compiled_model = core.compile_model(model=ppp_model, device_name="CPU")
        self.output_layer_ir = self.compiled_model.output(0)

    def set_kd_tree(self):
        grid = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        square_centers = warp(grid, self.keypoints)
        self.kd_tree = KDTree(square_centers)

    @staticmethod
    def _zero_king_scores(pred):
        for piece in ['white-king', 'black-king']:
            idx = 4 + CLASSES.index(piece)
            try:
                best_idx = pred[4:, idx].argmax()
            except ValueError:
                continue
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
        grid = (np.mgrid[0:8, 0:8].reshape(2, -1).T + 0.5) * SQUARE_SIZE
        square_centers = warp(grid, self.keypoints)
        return square_centers

    def _run_od(self, image):
        height, width = image.shape[:2]
        if self.input_width != width or self.input_height != height:
            self.input_width = width
            self.input_height = height
            self.set_compiled_model()
        pred = self.compiled_model([np.expand_dims(image, axis=0)])[self.output_layer_ir][0]
        pred = pred.transpose((1, 0))
        pred[:, 0] -= pred[:, 2] / 2
        pred[:, 1] -= pred[:, 3] / 2
        pred[:, 2] += pred[:, 0]
        pred[:, 3] += pred[:, 1]

        pred[:, [0, 2]] *= width / self.vino_width
        pred[:, [1, 3]] *= height / self.vino_height
        return pred

    def _filter_by_confidence(self, pred):
        scores = pred[:, 4:]
        best_scores_idx = np.argmax(scores, axis=1).reshape(-1, 1)
        best_scores = np.take_along_axis(scores, best_scores_idx, axis=1)
        mask = np.ravel(best_scores > self.conf_thres)
        pred = pred[mask]
        return pred

    def _filter_by_roi(self, pred):
        piece_centers = np.vstack([(pred[:, 0] + pred[:, 2]) / 2,
                                   pred[:, 3] - ((pred[:, 2] - pred[:, 0]) / 4)]).T
        mask = np.ones(len(piece_centers), dtype=bool)
        for i in range(4):
            v1 = self.keypoints[i-1] - self.keypoints[i]
            v2 = piece_centers - self.keypoints[i]
            cross_products = np.cross(v1, v2)
            mask &= cross_products < 0
        pred = pred[mask]
        piece_centers = piece_centers[mask]
        return pred, piece_centers

    def _filter_by_square(self, pred, piece_centers):
        _, idxs = self.kd_tree.query(piece_centers)
        matches = {}
        confs = np.max(pred[:, 4:], axis=1)
        for i in range(len(piece_centers)):
            if idxs[i] in matches and confs[i] > matches[idxs[i]][0]:
                continue
            matches[idxs[i]] = [confs[i], i]
        matches = {v[1]: k for k, v in matches.items()}

        mask = np.array(list(matches.keys()), dtype=int)
        pred = pred[mask]
        piece_centers = piece_centers[mask]
        squares = [get_square(i) for i in matches.values()]

        return pred, piece_centers, squares

    def _post_process_pred(self, pred):
        pred = self._filter_by_confidence(pred)
        pred, piece_centers = self._filter_by_roi(pred)
        pred, piece_centers, squares = self._filter_by_square(pred, piece_centers)
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

    def run(self, image: np.ndarray):
        pred = self._run_od(image)
        pred = self._post_process_pred(pred)
        return pred
