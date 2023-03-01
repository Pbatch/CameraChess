import os.path

from ultralytics import YOLO
import subprocess

from camera_chess.constants import DATA_DIR


def pt_to_onnx(pt_path, size):
    model = YOLO(pt_path)
    model.export(format='onnx',
                 imgsz=size,
                 simplify=True)


def pt_to_tfjs(pt_path, size):
    model = YOLO(pt_path)
    model.export(format='tfjs',
                 imgsz=size)


def onnx_to_openvino(onnx_path, size):
    subprocess.call(['mo',
                     '--input_model', onnx_path,
                     '--input_shape', f'[1,3,{size},{size}]',
                     '--data_type', 'FP16',
                     '--output_dir', 'models/'])


def openvino_quant(onnx_path):
    subprocess.call(["pot",
                     "-q", "default",
                     "-m", f'{os.path.splitext(onnx_path)[0]}.xml',
                     "-w", f'{os.path.splitext(onnx_path)[0]}.bin',
                     "--engine", "simplified",
                     "--data-source", os.path.join(DATA_DIR, 'google', 'images'),
                     "--output-dir", "models/INT8"])


def main():
    pt_path = 'models/480M.pt'
    onnx_path = 'models/480M.onnx'
    size = 480

    # pt_to_tfjs(pt_path, size)
    pt_to_onnx(pt_path, size)
    onnx_to_openvino(onnx_path, size)
    # openvino_quant(onnx_path)


if __name__ == '__main__':
    main()
