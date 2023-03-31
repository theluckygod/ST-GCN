import sys
import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np

sys.path.extend(['../'])
from graph.mediapipe_utils import LANDMARKS

max_body_true = 1
num_joint = 115
max_frame = 48
label_shape = 94477

OBJ_ITEMS = {"TV": 0, "after": 1, "airplane": 2, "all": 3, "alligator": 4, "animal": 5, "another": 6, "any": 7, "apple": 8, "arm": 9, "aunt": 10, "awake": 11, "backyard": 12, "bad": 13, "balloon": 14, "bath": 15, "because": 16, "bed": 17, "bedroom": 18, "bee": 19, "before": 20, "beside": 21, "better": 22, "bird": 23, "black": 24, "blow": 25, "blue": 26, "boat": 27, "book": 28, "boy": 29, "brother": 30, "brown": 31, "bug": 32, "bye": 33, "callonphone": 34, "can": 35, "car": 36, "carrot": 37, "cat": 38, "cereal": 39, "chair": 40, "cheek": 41, "child": 42, "chin": 43, "chocolate": 44, "clean": 45, "close": 46, "closet": 47, "cloud": 48, "clown": 49, "cow": 50, "cowboy": 51, "cry": 52, "cut": 53, "cute": 54, "dad": 55, "dance": 56, "dirty": 57, "dog": 58, "doll": 59, "donkey": 60, "down": 61, "drawer": 62, "drink": 63, "drop": 64, "dry": 65, "dryer": 66, "duck": 67, "ear": 68, "elephant": 69, "empty": 70, "every": 71, "eye": 72, "face": 73, "fall": 74, "farm": 75, "fast": 76, "feet": 77, "find": 78, "fine": 79, "finger": 80, "finish": 81, "fireman": 82, "first": 83, "fish": 84, "flag": 85, "flower": 86, "food": 87, "for": 88, "frenchfries": 89, "frog": 90, "garbage": 91, "gift": 92, "giraffe": 93, "girl": 94, "give": 95, "glasswindow": 96, "go": 97, "goose": 98, "grandma": 99, "grandpa": 100, "grass": 101, "green": 102, "gum": 103, "hair": 104, "happy": 105, "hat": 106, "hate": 107, "have": 108, "haveto": 109, "head": 110, "hear": 111, "helicopter": 112, "hello": 113, "hen": 114, "hesheit": 115, "hide": 116, "high": 117, "home": 118, "horse": 119, "hot": 120, "hungry": 121, "icecream": 122, "if": 123, "into": 124, "jacket": 125, "jeans": 126, "jump": 127, "kiss": 128, "kitty": 129, "lamp": 130, "later": 131, "like": 132, "lion": 133, "lips": 134, "listen": 135, "look": 136, "loud": 137, "mad": 138, "make": 139, "man": 140, "many": 141, "milk": 142, "minemy": 143, "mitten": 144, "mom": 145, "moon": 146, "morning": 147, "mouse": 148, "mouth": 149, "nap": 150, "napkin": 151, "night": 152, "no": 153, "noisy": 154, "nose": 155, "not": 156, "now": 157, "nuts": 158, "old": 159, "on": 160, "open": 161, "orange": 162, "outside": 163, "owie": 164, "owl": 165, "pajamas": 166, "pen": 167, "pencil": 168, "penny": 169, "person": 170, "pig": 171, "pizza": 172, "please": 173, "police": 174, "pool": 175, "potty": 176, "pretend": 177, "pretty": 178, "puppy": 179, "puzzle": 180, "quiet": 181, "radio": 182, "rain": 183, "read": 184, "red": 185, "refrigerator": 186, "ride": 187, "room": 188, "sad": 189, "same": 190, "say": 191, "scissors": 192, "see": 193, "shhh": 194, "shirt": 195, "shoe": 196, "shower": 197, "sick": 198, "sleep": 199, "sleepy": 200, "smile": 201, "snack": 202, "snow": 203, "stairs": 204, "stay": 205, "sticky": 206, "store": 207, "story": 208, "stuck": 209, "sun": 210, "table": 211, "talk": 212, "taste": 213, "thankyou": 214, "that": 215, "there": 216, "think": 217, "thirsty": 218, "tiger": 219, "time": 220, "tomorrow": 221, "tongue": 222, "tooth": 223, "toothbrush": 224, "touch": 225, "toy": 226, "tree": 227, "uncle": 228, "underwear": 229, "up": 230, "vacuum": 231, "wait": 232, "wake": 233, "water": 234, "wet": 235, "weus": 236, "where": 237, "white": 238, "who": 239, "why": 240, "will": 241, "wolf": 242, "yellow": 243, "yes": 244, "yesterday": 245, "yourself": 246, "yucky": 247, "zebra": 248, "zipper": 249}
s2p_map  = {k.lower():v for k,v in OBJ_ITEMS.items()}
p2s_map  = {v:k for k,v in OBJ_ITEMS.items()}
encoder  = lambda x: s2p_map.get(x.lower())
decoder  = lambda x: p2s_map.get(x)

def clean_skeleton(data0):
    frames_nansum = np.nanmean(data0, axis=[1,2])
    non_empty_frames_idxs = np.where(frames_nansum > 0)
    non_empty_frames_idxs = np.squeeze(non_empty_frames_idxs, axis=1)
    data = data0[non_empty_frames_idxs, :, :]
    return data

def extract_interest_frames(keypoint, max_frame): # keypoint.shape (T,V,C)
    T = keypoint.shape[0]

    keypoint = keypoint[:, LANDMARKS, :]

    if T >= max_frame:
        choiced_frames = list(range(0, T, round(T // max_frame)))
        choiced_frames = choiced_frames[:max_frame]
        keypoint = keypoint[choiced_frames, ...]
        T = keypoint.shape[0]
    
    if T < max_frame:
        keypoint = np.pad(keypoint, [(0, max_frame - T), (0, 0), (0, 0)], 'constant', constant_values=0)

    keypoint = np.moveaxis(keypoint, [0, 1, 2], [-2, -1, 0])[..., np.newaxis] # (T,V,C) -> (C,T,V,M)

    return keypoint

def gendata(data_path, out_path):
    fp = np.memmap('{}/{}_data_joint.npy'.format(out_path, "skeleton"), dtype='float32', mode='w+', shape=(label_shape,3, max_frame, num_joint, max_body_true))
    sample_name = []
    sample_label = []

    for filename in tqdm(os.listdir(data_path)):
        if not filename.endswith(".pickle"):
            continue

        with open(os.path.join(data_path, filename), "rb") as f:
            data_dict = pickle.load(f)

        for obj in tqdm(data_dict.values()):
            keypoint = obj["keypoint"]
            label = encoder(obj["label"])
            frame_dir = obj["frame_dir"]

            keypoint = clean_skeleton(keypoint)
            features = extract_interest_frames(keypoint, max_frame)

            fp[len(sample_name), :, 0:features.shape[1], :, :] = features
            sample_name.append(frame_dir)
            sample_label.append(label)

    with open('{}/{}_label.pkl'.format(out_path, "skeleton"), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASL Data Converter.')
    parser.add_argument('--data_path', default='../data/asl_data_raw/')
    parser.add_argument('--ignored_sample_path',
                        default=None)
    parser.add_argument('--out_folder', default='../data/asl_data/')

    arg = parser.parse_args()

    out_path = os.path.join(arg.out_folder)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gendata(
        arg.data_path,
        out_path)
