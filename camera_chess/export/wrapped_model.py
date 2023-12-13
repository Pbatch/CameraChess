import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import Detect, C2f


class WrappedModel(nn.Module):
    max_wh = 7680
    max_classes = 12

    def __init__(self, model, model_width=640, model_height=384, fill_colour=114):
        super().__init__()
        self.model = model
        self.model_width = model_width
        self.model_height = model_height
        self.fill_colour = fill_colour

        self.desired_ratio = self.model_height / self.model_width

    def _preprocess(self, x):
        # Change type (uint8 -> float32)
        x = x.float()

        # Resizing
        # NHWC -> (N, C, H*, W*)
        x = torch.permute(x, (0, 3, 1, 2))
        height, width = x.shape[2:]
        ratio = height / width
        if ratio > self.desired_ratio:
            height = self.model_height
            width = self.model_height / self.desired_ratio
        else:
            width = self.model_width
            height = self.model_width * self.desired_ratio

        x = F.interpolate(x,
                          size=(int(height), int(width)),
                          mode="bilinear",
                          align_corners=False,
                          )

        # Padding
        # (N, C, H*, W*) -> (N, C, image_size, image_size)
        height, width = x.shape[2:]
        dh = (self.model_height - height) / 2
        dw = (self.model_width - width) / 2
        padding = torch.tensor([dw - 0.1, dw + 0.1, dh - 0.1, dh + 0.1]).round().to(dtype=torch.int32).tolist()
        x = F.pad(x, padding, value=self.fill_colour)

        # Scaling
        x = x / 255

        return x, padding

    def _fix_bboxes(self, y, padding, width, height):
        # xywh -> xyxy
        y[..., 0] -= y[..., 2] / 2
        y[..., 1] -= y[..., 3] / 2
        y[..., 2] += y[..., 0]
        y[..., 3] += y[..., 1]

        # Unpad then unscale
        y[..., [0, 2]] -= padding[0]
        y[..., [1, 3]] -= padding[2]
        y[..., [0, 2]] *= width / (self.model_width - padding[0] - padding[1])
        y[..., [1, 3]] *= height / (self.model_height - padding[2] - padding[3])

        return y

    def forward(self, x):
        _, height, width, _ = x.shape
        x, padding = self._preprocess(x)
        y = self.model(x).transpose(2, 1)
        y = self._fix_bboxes(y, padding, width, height)
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
