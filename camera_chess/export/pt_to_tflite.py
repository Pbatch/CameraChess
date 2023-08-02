from ultralytics import YOLO


def main():
    model = YOLO('models/480S.pt')
    model.export(format='tflite', int8=True)


if __name__ == '__main__':
    main()