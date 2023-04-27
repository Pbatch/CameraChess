import os

import numpy as np
from PIL import Image
from openvino.runtime import AsyncInferQueue
from openvino.runtime import Tensor
from tqdm import tqdm

from camera_chess.detector import Detector, Detections
from camera_chess.state import State
from camera_chess.tracker.tracker import Tracker
from camera_chess.utils import load_video_config, clear_dir
from camera_chess.video import Video
from camera_chess.visualizer import Visualizer

error = ''
correct = 0


def callback(infer_request, info):
    detector, tracker, state, visualizer, image, frame, moves = info
    global error
    global correct

    pred = infer_request.get_output_tensor(0).data
    pred = detector.filter_by_roi(pred)
    detections = Detections(pred[:, :4], pred[:, 4], pred[:, 5].astype(int))
    tracks = tracker.update(detections)
    state.update(tracks)
    if state.change or np.random.random() < 1.0:
        image = Image.fromarray(image)
        image = visualizer.add_bboxes_from_tracks(image, tracks, detector.keypoints)
        image = visualizer.add_board(image, state)
        image.save(os.path.join('debug', f'{frame}.jpg'))
    if state.change:
        move_no = state.board.ply() - 1
        pred_move = state.last_move
        gt_move = moves[move_no]
        if pred_move == gt_move:
            correct += 1
        else:
            error += f'\nPredicted {pred_move} on move {move_no} at frame {frame} instead of {gt_move}'
            print(error)
            exit(1)


def main():
    dataset = 'peter/kasparov_immortal'
    video_config = load_video_config(dataset)
    video = Video(video_config, target_fps=4)
    video.save_start_image()
    detector = Detector(model_path='models/480S-quant.xml',
                        weights_path='models/480S-quant.bin',
                        keypoints=video.new_keypoints)
    tracker = Tracker(fps=video.target_fps,
                      keypoints=video.new_keypoints,
                      new_track_thresh=0.6,
                      track_high_thresh=0.6,
                      track_low_thresh=0.3)
    visualizer = Visualizer()
    state = State(video_config.fen)
    clear_dir('debug')

    infer_queue = AsyncInferQueue(detector.model, 2)
    infer_queue.set_callback(callback)
    for image, frame in tqdm(video, desc='Frame'):
        infer_queue.start_async({detector.input_layer_ir.any_name: Tensor(np.expand_dims(image, axis=0))},
                                (detector, tracker, state, visualizer, image, frame, video_config.moves))

    infer_queue.wait_all()
    if len(error):
        print(error)
    print(f'{correct}/{len(video_config.moves)} moves were tracked correctly')

    # visualizer.create_video('debug', fps=1)


if __name__ == '__main__':
    main()
