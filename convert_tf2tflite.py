from model.stgcn import Model
import tensorflow as tf
import numpy as np
import argparse
import os

tf.random.set_seed(42)
max_body_true = 1
num_joint = 115
max_frame = 48
batch_size = 8

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolutional Neural Network for Skeleton-Based Action Recognition')
    parser.add_argument(
        '--num-classes', type=int, default=250, help='number of classes in dataset')
    parser.add_argument(
        '--checkpoint-path',
        default="checkpoints/asl_data",
        help='folder to store model weights')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()

    num_classes     = arg.num_classes
    checkpoint_path = arg.checkpoint_path
    strategy        = tf.distribute.MirroredStrategy()


    with strategy.scope():
        model = Model(num_classes=num_classes)
        ckpt         = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()


    (N, C, T, V, M) = (batch_size, 3, max_frame, num_joint, max_body_true)
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
    model_path = os.path.join(checkpoint_path, 'model.tflite')
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


    interpreter = tf.lite.Interpreter(model_path=model_path)
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

    tflite_outputs = interpreter.get_tensor(output_details[0]['index'])

    print("input_distance:", tf.norm(tf_input_b1 - tflite_inputs))
    print("output_distance:", tf.norm(tf_output_b1 - tflite_outputs))


