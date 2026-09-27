import argparse

from camera_chess.utils import load_video_config
from camera_chess.video import Video


def main(dataset):
    video_config = load_video_config(dataset)
    video = Video(video_config, target_fps=8)
    video.save_start_image()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)

