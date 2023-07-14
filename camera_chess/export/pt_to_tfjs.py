from ultralytics import YOLO


def main():
    model = YOLO('models/480S_pruned.pt')
    model.export(format='tfjs')


if __name__ == '__main__':
    main()