import os
import shutil

from PIL import Image
from tqdm import tqdm

from camera_chess.classifier import Classifier
from camera_chess.state import State
from camera_chess.utils import load_video_config
from camera_chess.video import Video
from camera_chess.visualizer import Visualizer


def main():
    dataset = 'carlsen_vidit'
    video_config = load_video_config(dataset)
    video = Video(video_config, target_fps=4)
    classifier = Classifier(model_path='models/480S.onnx',
                            conf_thres=0.1,
                            keypoints=video.new_keypoints)
    visualizer = Visualizer()
    state = State(video.new_keypoints,
                  min_hits=2)
    if os.path.isdir('positions'):
        shutil.rmtree('positions')
    os.makedirs('positions')

    correct = 0
    with tqdm(total=len(video_config.moves)) as pbar:
        for image, frame in video:
            pred = classifier.run(image)
            state.update(pred)
            if state.change:
                image = Image.fromarray(image)
                image = visualizer.add_bboxes(image, pred)
                image = visualizer.add_board(image, state)
                image.save(os.path.join('positions', f'{frame}.jpg'))
                move_no = state.board.ply() - 1
                pred_move = state.last_move
                gt_move = video_config.moves[move_no]
                if pred_move == gt_move:
                    correct += 1
                else:
                    print(f'Predicted {pred_move} on move {move_no} '
                          f'at time {frame // video.fps} instead of {gt_move}')
                    state.debug(gt_move, pred)
                    break
                pbar.update(1)
    visualizer.create_video('positions', fps=video.target_fps)
    print(f'{correct}/{len(video_config.moves)} moves were tracked correctly')


if __name__ == '__main__':
    main()
