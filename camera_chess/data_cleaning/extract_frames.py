import os
from glob import glob
import cv2
from tqdm import tqdm


def main():
    target_fps = 2
    for path in tqdm(glob(os.path.join('data', 'youtube', 'videos', '*'))):
        frames_dir = os.path.join('data', 'youtube', 'frames',
                                  os.path.splitext(os.path.basename(path))[0])
        os.makedirs(frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        mod = fps / target_fps

        i = 0
        while True:
            success = cap.grab()
            if not success:
                break

            if i % mod == 0:
                _, image = cap.retrieve()
                cv2.imwrite(os.path.join(frames_dir, f'{i}.jpg'), image)
            i += 1


if __name__ == '__main__':
    main()
