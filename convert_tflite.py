import tensorflow  as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from model.stgcn import Model
import pandas as pd
import numpy as np

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


eps = 1e-3
max_body_true = 1
num_joint = 115
max_frame = 48
batch_size = 8
N_ROWS = 543
N_DIMS = 3
FACE_OFFSET = 0
LEFT_HAND_OFFSET = FACE_OFFSET + 468
POSE_OFFSET = LEFT_HAND_OFFSET + 21
RIGHT_HAND_OFFSET = POSE_OFFSET + 33

lip_landmarks = sorted([61, 185, 40, 39, 37,  0, 267, 269, 270, 409,
                 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])
left_hand_landmarks = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + 21))
right_hand_landmarks = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + 21))
pose_landmarks = list(range(POSE_OFFSET, POSE_OFFSET + 33))

LANDMARKS = lip_landmarks + left_hand_landmarks + pose_landmarks + right_hand_landmarks

num_classes     = 250
checkpoint_path = "checkpoints/asl_data"
model_path = 'model.tflite'
strategy        = tf.distribute.MirroredStrategy()


def clean_skeleton(data0):
    frames_nansum = tf.experimental.numpy.nanmean(data0, axis=[1,2])
    non_empty_frames_idxs = tf.where(frames_nansum > 0)
    non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
    data = tf.gather(data0, non_empty_frames_idxs, axis=0)
    
    nan_filter_z = tf.math.is_nan(data[:, :, 2])
    data = tf.concat([data[:, :, :2], tf.expand_dims(tf.where(nan_filter_z, 0.0, data[:, :, 2]), axis=2)], axis=2)
    data = tf.concat([data[:, :, :2], tf.expand_dims(tf.where(~nan_filter_z, 1.0, data[:, :, 2]), axis=2)], axis=2)
    data = tf.where(tf.math.is_nan(data), 0.0, data)
    return data

def extract_interest_frames(keypoint, max_frame):
    T = tf.shape(keypoint)[0]
    keypoint = tf.gather(keypoint, LANDMARKS, axis=1)

    if T >= max_frame:
        choiced_frames = tf.range(0, T, delta=tf.math.floordiv(T, max_frame), dtype=tf.int32)
        choiced_frames = choiced_frames[:max_frame]
        keypoint = tf.gather(keypoint, choiced_frames, axis=0)
        T  = tf.shape(keypoint)[0]

    if T < max_frame:
        pad_width = [(0, max_frame - T), (0, 0), (0, 0)]
        keypoint = tf.pad(keypoint, pad_width, 'CONSTANT')

    keypoint = tf.reshape(keypoint, (max_frame, tf.shape(keypoint)[1], tf.shape(keypoint)[2]))
    keypoint = tf.transpose(keypoint, [2, 0, 1]) # (T,V,C) -> (C,T,V)
    keypoint = tf.expand_dims(keypoint, axis=-1) # (C,T,V) -> (C,T,V,M)

    return keypoint

class Preprocessing(tf.keras.layers.Layer):
    def __init__(self):
        super(Preprocessing, self).__init__()
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def call(self, inputs):
        keypoint = clean_skeleton(inputs)
        keypoint = extract_interest_frames(keypoint, max_frame)
        keypoint = tf.expand_dims(keypoint, axis=0)
        return keypoint


class FinalModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prep = Preprocessing()
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        keypoint = self.prep(inputs)
        tf_output = self.model(keypoint, training=False)        
        return {'outputs': tf_output}

class WrapModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, 48, 115, 1], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        tf_output = self.model(inputs, training=False)        
        return {'outputs': tf_output}

if __name__ == "__main__":

    # with strategy.scope():
    #     model = Model(num_classes=num_classes)
    #     ckpt  = tf.train.Checkpoint(model=model)
    #     ckpt_manager = tf.train.CheckpointManager(ckpt,
    #                                             checkpoint_path,
    #                                             max_to_keep=5)
    #     ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    model = Model(num_classes=num_classes)
    final_model = FinalModel(model=model)
    # final_model = WrapModel(model=model)

    keypoint = load_relevant_data_subset("/mnt/kaggle/asl_data/train_landmark_files/16069/100015657.parquet")
    # keypoint = tf.random.normal(shape=(1, 3, 48, 115, 1), dtype=tf.dtypes.float32)
    print("=============keypoint", keypoint.shape)

    print(keypoint.shape)
    result_tf = final_model(keypoint)

    sign = np.argmax(result_tf["outputs"])

    print("\ntf_pred", sign)

    ## convert
    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(model_path, 'wb') as f:
        f.write(tflite_model)

    ## test predict
    interpreter = tf.lite.Interpreter(model_path=model_path)

    print("tflite input", interpreter.get_input_details()[0])
    print("tflite output", interpreter.get_output_details()[0])

    prediction_fn = interpreter.get_signature_runner("serving_default")
    tflite_outputs = prediction_fn(inputs=keypoint)
    tflite_sign = np.argmax(tflite_outputs["outputs"])

    print("\ntflite_pred", tflite_sign)
    print("output_distance:", tf.norm(result_tf["outputs"] - tflite_outputs["outputs"], ord='euclidean'))
