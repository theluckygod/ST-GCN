from model.stgcn import Model
import tensorflow as tf
from tqdm import tqdm
import argparse

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from train import get_dataset

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.random.set_seed(42)


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolutional Neural Network for Skeleton-Based Action Recognition')
    parser.add_argument(
        '--num-classes', type=int, default=250, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=8, help='training batch size')
    parser.add_argument(
        '--checkpoint-path',
        default="checkpoints/asl_data",
        help='folder to store model weights')
    parser.add_argument(
        '--test-data-path',
        default="data/asl_data/asl_data_test",
        help='path to folder with testing dataset tfrecord files')
    parser.add_argument(
        '--gpus',
        default=None,
        nargs='+',
        help='list of gpus to use for training, eg: "/gpu:0" "/gpu:1"')

    return parser


'''
test_step: gets model prediction for given samples
Args:
  features: tensor with features
'''
@tf.function
def test_step(features):
    logits = model(features, training=False)
    return tf.nn.softmax(logits)


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()

    num_classes     = arg.num_classes
    checkpoint_path = arg.checkpoint_path
    test_data_path  = arg.test_data_path
    batch_size      = arg.batch_size
    gpus            = arg.gpus
    strategy        = tf.distribute.MirroredStrategy(arg.gpus)
    global_batch_size = arg.batch_size*strategy.num_replicas_in_sync
    arg.gpus        = strategy.num_replicas_in_sync


    '''
    Get tf.dataset objects for training and testing data
    Data shape: features - batch_size, 3, 300, 25, 2
                labels   - batch_size, num_classes
    '''
    test_data = get_dataset(test_data_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            drop_remainder=False,
                            shuffle=False)

    with strategy.scope():
        model = Model(num_classes=num_classes)
        ckpt         = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    test_acc             = tf.keras.metrics.CategoricalAccuracy(name='test_acc')

    print("Testing: ")
    for features, labels in tqdm(test_data):
        y_pred = test_step(features)
        test_acc(labels, y_pred)
    print("test_acc:      ", test_acc.result())
