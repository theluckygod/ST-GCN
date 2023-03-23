import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import random
random.seed(10)

from gen_joint_data import max_body_true, num_joint, max_frame, label_shape


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(features, label):
    feature = {
        'features' : _bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
        'label'     : _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def split_train_val(data, labels, ratio=0.15):
    class_dict = {}
    for idx, label in enumerate(labels):
        tmp = class_dict.get(label, [])
        tmp.append(idx)
        class_dict[label] = tmp

    train_selected = []
    val_selected = []
    train_y = []
    val_y = []
    for key in class_dict:
        label_list = class_dict[key]
        val_labels_tmp = random.sample(label_list, round(len(label_list) * ratio))
        val_selected += val_labels_tmp
        
        for l in label_list:
            if l not in val_labels_tmp:
                train_selected.append(l)

        train_y += [key] * (len(label_list) - len(val_labels_tmp))
        val_y += [key] * len(val_labels_tmp)

    train_x = data[train_selected, ...]
    val_x = data[val_selected, ...]


    return train_x, np.array(train_y), val_x, np.array(val_y)

def save_data(data, labels, data_path, dest_folder, num_shards, shuffle):
    if len(labels) != len(data):
        print("Data and label lengths didn't match!")
        print("Data size: {} | Label Size: {}".format(data.shape, labels.shape))
        raise ValueError("Data and label lengths didn't match!")

    print("Data shape:", data.shape)
    if shuffle:
        p = np.random.permutation(len(labels))
        labels = labels[p]
        data = data[p]

    dest_folder = Path(dest_folder)
    if not (dest_folder.exists()):
        os.mkdir(dest_folder)

    step = len(labels)//num_shards
    for shard in tqdm(range(num_shards)):
        tfrecord_data_path = os.path.join(dest_folder, data_path.name.split(".")[0]+"-"+str(shard)+".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
            for i in range(shard*step, (shard*step)+step if shard < num_shards-1 else len(labels)):
                writer.write(serialize_example(data[i], labels[i]))

def gen_tfrecord_data(num_shards, label_path, data_path, dest_folder, shuffle):
    label_path = Path(label_path)
    if not (label_path.exists()):
        print('Label file does not exist')
        return

    data_path = Path(data_path)
    if not (data_path.exists()):
        print('Data file does not exist')
        return

    try:
        with open(label_path) as f:
            _, labels = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            _, labels = pickle.load(f, encoding='latin1')

    # Datashape: Total_samples, 3, 300, 25, 2
    data   = np.memmap(data_path, dtype='float32', mode='r', shape=(label_shape, 3, max_frame, num_joint, max_body_true))
    train_x, train_y, val_x, val_y = split_train_val(data, labels)
    save_data(train_x, train_y, data_path, dest_folder + "_train", num_shards, shuffle)
    save_data(val_x, val_y, data_path, dest_folder + "_test", num_shards, shuffle)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data TFRecord Converter')
    parser.add_argument('--num-shards',
                        type=int,
                        default=1,
                        help='number of files to split dataset into')
    parser.add_argument('--label-path',
                        # required=True,
                        default="../data/asl_data/skeleton_label.pkl",
                        help='path to pkl file with labels')
    parser.add_argument('--shuffle',
                        # required=True,
                        default=True,
                        help='setting it to True will shuffle the labels and data together')
    parser.add_argument('--data-path',
                        # required=True,
                        default="../data/asl_data/skeleton_data_joint.npy",
                        help='path to npy file with data')
    parser.add_argument('--dest-folder',
                        # required=True,
                        default="../data/asl_data/asl_data",
                        help='path to folder in which tfrecords will be stored')
    arg = parser.parse_args()

    gen_tfrecord_data(arg.num_shards,
                      arg.label_path,
                      arg.data_path,
                      arg.dest_folder,
                      arg.shuffle)
