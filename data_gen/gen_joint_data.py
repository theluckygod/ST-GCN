import sys
import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

max_body_true = 1
num_joint = 115
max_frame = 48


def extract_interest_frames(keypoint, max_frame):
    pass

def gendata(data_path, out_path, benchmark):
    sample_label = []
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    for filename in os.listdir(data_path):
        if not filename.endswith(".pickle"):
            continue

        with open(os.path.join(data_path, filename), "rb") as f:
            data_dict = pickle.load(f)

        for obj in data_dict.values():
            keypoint = obj["keypoint"]
            label = obj["label"]
            total_frames = obj["total_frames"]

            features = extract_interest_frames(keypoint, max_frame)

            fp[i, :, 0:features.shape[1], :, :] = features

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASL Data Converter.')
    parser.add_argument('--data_path', default='../data/asl_data_raw/')
    parser.add_argument('--ignored_sample_path',
                        default=None)
    parser.add_argument('--out_folder', default='../data/asl_data/')

    benchmark = 'asl_data'
    arg = parser.parse_args()

    out_path = os.path.join(arg.out_folder, benchmark)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gendata(
        arg.data_path,
        out_path,
        benchmark=benchmark)
