# -*- encoding=utf-8 -*-
# import tensorflow as tf
import tflite_runtime.interpreter as tflite

import numpy as np

PATH_TO_TENSORFLOW_MODEL = 'models/face_mask_detection.pb'
EDGETPU_SHARED_LIB = "libedgetpu.so.1"

def load_tf_model(tf_model_path):
    # Load the TFLite model and allocate tensors.
    model_file, *device = tf_model_path.split('@')
    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
           tflite.load_delegate(EDGETPU_SHARED_LIB,
                                {'device': device[0]} if device else {})
         ]
    )

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def tf_inference(interpreter, input_details, output_details, img_arr):
    img_arr = img_arr.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], [np.reshape(img_arr, (260, 260, 3))])
    interpreter.invoke()
    bboxes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[1]['index'])

    return bboxes, scores

