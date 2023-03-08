import os

from PIL import Image
from tqdm import tqdm

from camera_chess.detector import Detector
from camera_chess.state import State
from camera_chess.tracker import Tracker
from camera_chess.utils import load_video_config, clear_dir
from camera_chess.video import Video
from camera_chess.visualizer import Visualizer


def main():
    dataset = 'youtube/master_andrea'
    video_config = load_video_config(dataset)
    video = Video(video_config, target_fps=8)
    video.save_start_image()
    detector = Detector(model_path='models/480S-sim-quant.xml',
                        weights_path='models/480S-sim-quant.bin',
                        conf_thres=0.1,
                        keypoints=video.new_keypoints)
    tracker = Tracker(fps=video.target_fps,
                      keypoints=video.new_keypoints,
                      track_low_thresh=detector.conf_thres)
    visualizer = Visualizer()
    state = State(video_config.fen)
    clear_dir('debug')

    correct = 0
    error = None
    with tqdm(total=len(video_config.moves), desc='Move') as pbar:
        for image, frame in tqdm(video, desc='Frame'):
            detections = detector.run(image)
            tracks = tracker.update(detections)
            state.update(tracks)
            if state.change:
                pbar.update(1)
                image = Image.fromarray(image)
                image = visualizer.add_bboxes(image, tracks, video.new_keypoints)
                image = visualizer.add_board(image, state)
                image.save(os.path.join('debug', f'{frame}.jpg'))
                move_no = state.board.ply() - 1
                pred_move = state.last_move
                gt_move = video_config.moves[move_no]
                if pred_move == gt_move:
                    correct += 1
                else:
                    error = f'Predicted {pred_move} on move {move_no} at frame {frame} instead of {gt_move}'
                    break
                if move_no == len(video_config.moves) - 1:
                    break
    if error is not None:
        print(error)
    print(f'{correct}/{len(video_config.moves)} moves were tracked correctly')

    visualizer.create_video('debug', fps=1)


if __name__ == '__main__':
    main()
