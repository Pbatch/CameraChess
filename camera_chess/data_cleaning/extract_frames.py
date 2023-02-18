import os
from glob import glob
import cv2
from tqdm import tqdm


def main():
    target_fps = 4
    for path in tqdm(glob(os.path.join('data', 'youtube', 'videos', '*'))):
        if os.path.basename(path) != 'It_s_Blitz_says_Hikaru_Nakamura_after_his_game_against_Nihal_Sarin_World_Blitz_2022.webm':
            continue
        print(path)
        frames_dir = os.path.join('data', 'youtube', 'frames',
                                  os.path.splitext(os.path.basename(path))[0])
        os.makedirs(frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        mod = int(round(fps / target_fps))
        print(mod)

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
