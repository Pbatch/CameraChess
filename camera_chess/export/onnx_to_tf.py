import os

import onnx
from onnx_tf.backend import prepare


def onnx_to_tf(onnx_path):
    """
    You must use opset 12 when running pt_to_onnx for this to work
    """
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    tf_dir = os.path.splitext(onnx_path)[0]
    tf_rep.export_graph(tf_dir)


def main():
    onnx_path = 'models/480S.onnx'
    onnx_to_tf(onnx_path)


if __name__ == '__main__':
    main()
