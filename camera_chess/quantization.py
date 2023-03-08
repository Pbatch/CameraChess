import os
from glob import glob

import cv2
import nncf
import numpy as np
from PIL import Image

from openvino.runtime import Core, serialize
from torch.utils.data import Dataset, DataLoader

from camera_chess.constants import DATA_DIR


class ValDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.image_paths = sorted(glob(os.path.join(DATA_DIR, 'yolo', 'val', 'images', '*.jpg')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.array(Image.open(image_path).convert('RGB'))
        image = cv2.resize(image, (480, 480), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        image = image / 255
        image = image.astype(np.float32)

        return image


def default_quantization(model_path):
    model = Core().read_model(model_path)

    dataset = ValDataset()
    dataloader = DataLoader(dataset)
    calibration_dataset = nncf.Dataset(dataloader)

    ignored_scope = nncf.IgnoredScope(
        types=["Multiply", "Subtract", "Sigmoid"],
        names=[
            "/model.22/dfl/conv/Conv",
            "/model.22/Add",
            "/model.22/Add_1",
            "/model.22/Add_2",
            "/model.22/Add_3",
            "/model.22/Add_4",
            "/model.22/Add_5",
            "/model.22/Add_6",
            "/model.22/Add_7",
            "/model.22/Add_8",
            "/model.22/Add_9",
            "/model.22/Add_10"
        ]
    )
    quantized_model = nncf.quantize(model=model,
                                    calibration_dataset=calibration_dataset,
                                    preset=nncf.QuantizationPreset.MIXED,
                                    target_device=nncf.TargetDevice.CPU,
                                    ignored_scope=ignored_scope)

    save_path = model_path.replace('.xml', '-quant.xml')
    print(f'Saving quantized model to {save_path}')
    serialize(quantized_model, save_path)


def main():
    model_path = 'models/480S-sim.xml'
    default_quantization(model_path)


if __name__ == '__main__':
    main()
