import os

import numpy as np
import tensorflow as tf


def tf_to_tflite(tf_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(tf_dir, 'converted_model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model_path


def main():
    # tf_dir = 'models/480S'
    # model_path = tf_to_tflite(tf_dir)

    model_path = 'models/480S/converted_model.tflite'
    test(model_path)


def test(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], np.zeros(shape=(1, 400, 500, 3), dtype=np.uint8))
    output = interpreter.invoke()
    print(output)


if __name__ == '__main__':
    main()
