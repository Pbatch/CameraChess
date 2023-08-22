import argparse

from ultralytics import YOLO


def main(model_path, model_format):
    model = YOLO(model=model_path)

    kwargs = {}
    model.export(format=model_format,
                 **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--format', '-f', type=str, choices=['tfjs', 'onnx'])
    args = parser.parse_args()
    main(args.path, args.format)
