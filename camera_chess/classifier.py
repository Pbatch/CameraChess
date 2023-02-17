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
    def __init__(self, model_path, keypoints, conf_thres=0.2, iou_thres=0.4):
        self.model_path = model_path
        self.keypoints = keypoints
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.sess = onnxruntime.InferenceSession(self.model_path)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.onnx_height, self.onnx_width = self.sess.get_inputs()[0].shape[-2:]
        self.square_centers = self._get_square_centers()
        self.kd_tree = KDTree(self.square_centers)
        self.roi = Polygon(self.keypoints)

    @staticmethod
    def _xywh_to_xyxy(boxes):
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    @staticmethod
    def _nms(boxes, scores, thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0]  # pick maximum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

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

    def nms(self, prediction):
        """
        Runs Non-Maximum Suppression (NMS) on inference results
        """
        prediction = prediction.transpose((0, 2, 1))
        output = [np.zeros((0, 6))] * len(prediction)
        for i in range(len(prediction)):
            x = prediction[i]

            # Calculate the best scores
            scores = x[:, 4:]
            best_scores_idx = np.argmax(scores, axis=1).reshape(-1, 1)
            best_scores = np.take_along_axis(scores, best_scores_idx, axis=1)

            # Mask out predictions below the confidence threshold
            mask = np.ravel(best_scores > self.conf_thres)
            best_scores = best_scores[mask]
            best_scores_idx = best_scores_idx[mask]

            # Convert the xywh of each box to xyxy inplace
            boxes = x[mask, :4]
            self._xywh_to_xyxy(boxes)

            # Work out which boxes to keep
            keep = self._nms(boxes, np.ravel(best_scores), self.iou_thres)

            # Keep only the best class
            best = np.hstack([boxes[keep], best_scores[keep], best_scores_idx[keep]])

            output[i] = best
        return output

    def _run_od(self, np_image, width, height):
        pred = self.sess.run([self.output_name], {self.input_name: np_image})[0]
        pred[:, [0, 2], :] *= width / self.onnx_width
        pred[:, [1, 3], :] *= height / self.onnx_height
        pred = np.array(self.nms(pred)[0])

        return pred

    def _post_process_pred(self, pred):
        piece_centers = [[(p[0] + p[2]) / 2,
                          p[3] - ((p[2] - p[0]) / 4)]
                         for p in pred]
        distances, idxs = self.kd_tree.query(piece_centers)
        matches = {}
        for i in range(len(piece_centers)):
            if idxs[i] in matches and distances[i] > matches[idxs[i]][0]:
                continue
            matches[idxs[i]] = [distances[i], i]
        matches = {v[1]: k for k, v in matches.items()}

        clean_pred = []
        for i, (p, center) in enumerate(zip(pred, piece_centers)):
            if i not in matches:
                continue
            polygon = Polygon([(p[0], p[1]),
                               (p[0], p[3]),
                               (p[2], p[3]),
                               (p[2], p[1])])
            if not polygon.intersects(self.roi):
                continue
            bbox = np.round(p[:4]).astype(int)
            confidence = p[4]
            piece = CLASSES[int(p[5])]
            square = get_square(matches[i])
            clean_pred.append(classification(bbox, confidence, piece, center, square))

        return clean_pred

    def run(self, image, keypoints):
        np_image = self._preprocess_image(image)
        pred = self._run_od(np_image, image.width, image.height)
        pred = self._post_process_pred(pred)
        return pred
