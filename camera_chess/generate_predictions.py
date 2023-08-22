from ultralytics import YOLO


def main():
    model = YOLO('models/480N.pt')
    model.predict('data/yolo/train/images/*.jpg',
                  device='cuda',
                  half=True,
                  save_txt=True,
                  save_conf=True)


if __name__ == '__main__':
    main()
