import tensorflow as tf
import numpy as np
from model.stgcn import Model

BATCH_SIZE = 4
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300
# (N, C, T, V, M) = (BATCH_SIZE, 3, max_frame, num_joint, max_body_true) # (BATCH_SIZE, 1, 48, 110, 1)
(N, C, T, V, M) = (BATCH_SIZE, 3, 48, 115, 1)

model = Model(num_classes=250) 
inputs = np.random.rand(N, C, T, V, M)

tf_output = model(inputs, training=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.experimental_new_converter=True
converter.allow_custom_ops=True

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

tf_input_b1 = inputs[:1, ...]
tf_output_b1 = tf_output[:1, :]
tflite_inputs = tf_input_b1.astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], tflite_inputs)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_outputs = interpreter.get_tensor(output_details[0]['index'])

print("input_distance:", tf.norm(tf_input_b1 - tflite_inputs))
print("output_distance:", tf.norm(tf_output_b1 - tflite_outputs))


