import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from ultralytics.nn.modules import Detect, C2f


class WrappedModel(nn.Module):
    max_wh = 7680
    max_classes = 12

    def __init__(self, model, image_size=480, fill_colour=114, conf_thres=0.1, iou_thres=0.4):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.fill_colour = fill_colour
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _preprocess(self, x):
        # Change type (uint8 -> float32)
        x = x.float()

        # Resizing
        # NHWC -> (N, C, H*, W*)
        x = torch.permute(x, (0, 3, 1, 2))
        height, width = x.shape[2:]
        ratio = height / width
        if ratio > 1:
            height = self.image_size
            width = self.image_size / ratio
        else:
            width = self.image_size
            height = self.image_size * ratio

        x = F.interpolate(x,
                          size=(int(height), int(width)),
                          mode="bilinear",
                          align_corners=False,
                          )

        # Padding
        # (N, C, H*, W*) -> (N, C, image_size, image_size)
        height, width = x.shape[2:]
        dh = (self.image_size - height) / 2
        dw = (self.image_size - width) / 2
        padding = torch.tensor([dw - 0.1, dw + 0.1, dh - 0.1, dh + 0.1]).round().to(dtype=torch.int32).tolist()
        x = F.pad(x, padding, value=self.fill_colour)

        # Scaling
        x = x / 255

        return x, padding

    def _fix_bboxes(self, y, padding, width, height):
        # xywh -> xyxy
        y[:, 0] -= y[:, 2] / 2
        y[:, 1] -= y[:, 3] / 2
        y[:, 2] += y[:, 0]
        y[:, 3] += y[:, 1]

        # Unpad then unscale
        y[:, [0, 2]] -= padding[0]
        y[:, [1, 3]] -= padding[2]
        y[:, [0, 2]] *= width / (self.image_size - padding[0] - padding[1])
        y[:, [1, 3]] *= height / (self.image_size - padding[2] - padding[3])

        return y

    def _nms(self, y):
        y = y.transpose(2, 1)

        # Multi-label
        batch_idx, box_idx, class_idx = (y[..., 4:] > self.conf_thres).nonzero(as_tuple=False).T
        boxes = y[batch_idx, box_idx, :4]
        scores = y[batch_idx, box_idx, 4 + class_idx]

        shifted_boxes = boxes + class_idx[:, None] * self.max_wh + batch_idx[:, None] * self.max_wh * self.max_classes
        keep = torchvision.ops.nms(shifted_boxes, scores, self.iou_thres)

        scores = torch.unsqueeze(scores, dim=1)
        class_idx = torch.unsqueeze(class_idx, dim=1)
        batch_idx = torch.unsqueeze(batch_idx, dim=1)
        result = torch.concatenate([batch_idx[keep], boxes[keep], scores[keep], class_idx[keep]], dim=1)

        return result

    def forward(self, x):
        _, height, width, _ = x.shape
        x, padding = self._preprocess(x)
        y = self.model(x)
        y = self._fix_bboxes(y, padding, width, height)
        y = self._nms(y)
        return y


def load_model(model_path, device):
    model = torch.load(model_path, map_location='cpu')['model']
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.dynamic = True
            m.export = True
            m.format = 'onnx'
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    wrapped_model = WrappedModel(model)
    wrapped_model.to(device)
    return wrapped_model
