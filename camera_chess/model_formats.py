import subprocess

from ultralytics import YOLO


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


def main():
    pt_path = 'models/480M.pt'
    onnx_path = 'models/480S.onnx'
    size = 480

    # pt_to_tfjs(pt_path, size)
    # pt_to_onnx(pt_path, size)
    # onnx_to_openvino(onnx_path, size)


if __name__ == '__main__':
    main()
