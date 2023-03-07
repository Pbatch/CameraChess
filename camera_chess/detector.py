from collections import namedtuple

import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type

Detections = namedtuple("Detections", "xyxy conf cls")


class Detector:
    def __init__(self, model_path, keypoints, weights_path='', conf_thres=0.1, iou_thres=0.4):
        self.model_path = model_path
        self.keypoints = keypoints
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model = Core().read_model(model=self.model_path,
                                       weights=self.weights_path)
        self.vino_height = 480
        self.vino_width = 480

        self.compiled_model = None
        self.output_layer_ir = None
        self.input_width = 0
        self.input_height = 0

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
        return pred

    def _set_compiled_model(self):
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
        self.compiled_model = Core().compile_model(model=ppp_model, device_name="CPU")
        self.output_layer_ir = self.compiled_model.output(0)

    @staticmethod
    def _nms(boxes, scores, thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def nms(self, prediction):
        # Discard predictions below the confidence threshold
        prediction = prediction.transpose((1, 0))
        box_idx, class_idx = (prediction[:, 4:] > self.conf_thres).nonzero()
        scores = prediction[box_idx, 4 + class_idx]

        # Convert the xywh of each box to xyxy inplace
        boxes = prediction[box_idx, :4]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Run NMS
        shifted_boxes = boxes + class_idx[:, None] * 10000
        keep = self._nms(shifted_boxes, scores, self.iou_thres)
        output = np.hstack([boxes[keep], scores[keep, None], class_idx[keep, None]])

        return output

    def run(self, image: np.ndarray):
        height, width = image.shape[:2]
        if self.input_width != width or self.input_height != height:
            self.input_width = width
            self.input_height = height
            self._set_compiled_model()
        pred = self.compiled_model([np.expand_dims(image, axis=0)])[self.output_layer_ir][0]
        pred = self.nms(pred)
        pred[:, [0, 2]] *= width / self.vino_width
        pred[:, [1, 3]] *= height / self.vino_height
        pred = self._filter_by_roi(pred)

        detections = Detections(pred[:, :4], pred[:, 4], pred[:, 5].astype(int))
        return detections
