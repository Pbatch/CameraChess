import os
from collections import namedtuple
from glob import glob

import numpy as np
import onnxruntime
from PIL import Image
from icecream import ic

from camera_chess.constants import CLASSES
from camera_chess.visualizer import Visualizer

classification = namedtuple("Classification", "bbox conf piece")


class Classifier:
    def __init__(self, model_path):
        self.model_path = model_path

        self.sess = onnxruntime.InferenceSession(self.model_path)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.onnx_height, self.onnx_width = self.sess.get_inputs()[0].shape[-2:]

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

    def _preprocess_image(self, image):
        image = image.resize((self.onnx_width, self.onnx_height), Image.BICUBIC)
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        image = image / 255
        return image

    def nms(self, prediction, conf_thres=0.35, iou_thres=0.45):
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
            mask = np.ravel(best_scores > conf_thres)
            best_scores = best_scores[mask]
            best_scores_idx = best_scores_idx[mask]

            # Convert the xywh of each box to xyxy inplace
            boxes = x[mask, :4]
            self._xywh_to_xyxy(boxes)

            # Work out which boxes to keep
            keep = self._nms(boxes, np.ravel(best_scores), iou_thres)

            # Keep only the best class
            best = np.hstack([boxes[keep], best_scores[keep], best_scores_idx[keep]])

            output[i] = best
        return output

    def run(self, image):
        height, width = image.height, image.width
        np_image = self._preprocess_image(image)
        pred = self.sess.run([self.output_name], {self.input_name: np_image})[0]
        pred = np.array(self.nms(pred)[0])
        pred[:, [0, 2]] *= width / self.onnx_width
        pred[:, [1, 3]] *= height / self.onnx_height
        pred = [classification(np.round(p[:4]).astype(int), p[4], CLASSES[int(p[5])])
                for p in pred]
        return pred


def main():
    classifier = Classifier(model_path='data/best.onnx')
    visualizer = Visualizer()
    for path in sorted(glob('data/hikaru/crops/*.jpg'),
                       key=lambda x: int(os.path.basename(x).replace('.jpg', ''))):
        image = Image.open(path).convert('RGB')
        pred = classifier.run(image)
        visualizer.plot_bboxes(image, pred)


if __name__ == '__main__':
    main()
