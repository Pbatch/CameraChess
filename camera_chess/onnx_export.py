import os.path

from ultralytics import YOLO
import subprocess


def pt_to_onnx(pt_path):
    model = YOLO(pt_path)
    model.export(format='onnx',
                 imgsz=480,
                 simplify=True,
                 half=True)


def quantize(onnx_path):
    subprocess.call(['mo',
                     '--input_model', onnx_path,
                     '--input_shape', '[1,3,480,480]',
                     '--data_type', 'FP16',
                     '--output_dir', 'models/'])
    subprocess.call(["pot",
                     "-q", "default",
                     "-m", f'{os.path.splitext(onnx_path)[0]}.xml',
                     "-w", f'{os.path.splitext(onnx_path)[0]}.bin',
                     "--engine", "simplified",
                     "--data-source", "data/google/images",
                     "--output-dir", "models/INT8"])


def main():
    onnx_path = 'models/480S.onnx'
    quantize(onnx_path)


if __name__ == '__main__':
    main()
