import os.path

from ultralytics import YOLO
import subprocess

import constants


def pt_to_onnx(pt_path, size):
    model = YOLO(pt_path)
    model.export(format='onnx',
                 imgsz=size,
                 simplify=True,
                 half=True)


def quantize(onnx_path, size):
    subprocess.call(['mo',
                     '--input_model', onnx_path,
                     '--input_shape', f'[1,3,{size},{size}]',
                     '--data_type', 'FP16',
                     '--output_dir', 'models/'])
    subprocess.call(["pot",
                     "-q", "default",
                     "-m", f'{os.path.splitext(onnx_path)[0]}.xml',
                     "-w", f'{os.path.splitext(onnx_path)[0]}.bin',
                     "--engine", "simplified",
                     "--data-source", f"{constants.DATA_DIR}/google/images",
                     "--output-dir", "models/INT8"])


def main():
    pt_path = 'models/480S.pt'
    onnx_path = 'models/480S.onnx'
    size = 480

    pt_to_onnx(pt_path, size)
    quantize(onnx_path, size)


if __name__ == '__main__':
    main()
