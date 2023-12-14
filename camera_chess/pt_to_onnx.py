import os

from ultralytics import YOLO

from camera_chess.constants import MODEL_DIR


def main():
    model = YOLO(os.path.join(MODEL_DIR, '640S.pt'))
    # imgsz is HxW
    model.export(format='onnx',
                 imgsz=[384, 640],
                 simplify=True,
                 device=0)


if __name__ == '__main__':
    main()
