import subprocess


def onnx_to_openvino(onnx_path):
    subprocess.call(['mo',
                     '--input_model', onnx_path,
                     '--data_type', 'FP16',
                     '--output_dir', 'models/'])


def main():
    onnx_path = 'models/480S.onnx'
    onnx_to_openvino(onnx_path)


if __name__ == '__main__':
    main()
