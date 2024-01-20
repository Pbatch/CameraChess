import argparse
import os
import shutil

from ultralytics import YOLO

from camera_chess.constants import MODEL_DIR


def main(model_id, width, height):
    model = YOLO(os.path.join(MODEL_DIR, f'{model_id}.pt'))
    # imgsz is HxW
    model.export(format='onnx',
                 imgsz=[height, width],
                 simplify=True,
                 device=0)

    default_save_path = os.path.join(MODEL_DIR, f'{model_id}.onnx')
    new_save_path = os.path.join(MODEL_DIR, f'{model_id}_{width}x{height}.onnx')
    shutil.move(default_save_path, new_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_id', type=str, required=True)
    parser.add_argument('-mw', '--width', type=int, default=640)
    parser.add_argument('-mh', '--height', type=int, default=384)
    args = parser.parse_args()
    main(args.model_id, args.width, args.height)
