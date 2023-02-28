import ast

import gradio as gr
import numpy as np

from camera_chess.classifier import Classifier
from camera_chess.visualizer import Visualizer


def get_board(image, keypoints):
    keypoints = np.array(ast.literal_eval(keypoints),
                         dtype=np.float32)
    classifier = Classifier('models/480S.onnx',
                            keypoints=keypoints)
    visualizer = Visualizer()
    pred = classifier.run(image)
    image = visualizer.add_bboxes(image, pred)
    return image


def main():
    inputs = [gr.Image(type="pil"), gr.Text()]
    outputs = gr.Image(type="pil")

    examples = [['data/hikaru_sarin/images/19.jpg', "[[177, 476], [16, 171], [574, 73], [884, 318]]"]]
    demo = gr.Interface(fn=get_board,
                        inputs=inputs,
                        outputs=outputs,
                        examples=examples)
    demo.launch(share=True)


if __name__ == '__main__':
    main()
