import os
import pathlib
import random
import shutil
import keras
from keras.applications.efficientnet import preprocess_input

splits = (0.8, 0.1, 0.1)
src_dir = "data/Corn_raw"
dest_dir = "data/Corn_processed"

def split_data(src_dir, dest_dir, splits):
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    for cls in classes:
        cls_src = os.path.join(src_dir, cls)
        cls_dest = os.path.join(dest_dir, cls)
        os.makedirs(os.path.join(cls_dest, "train"), exist_ok=True)
        os.makedirs(os.path.join(cls_dest, "val"), exist_ok=True)
        os.makedirs(os.path.join(cls_dest, "test"), exist_ok=True)

        files = [f for f in os.listdir(cls_src) if os.path.isfile(os.path.join(cls_src, f))]
        random.shuffle(files)
        train_split = int(len(files) * splits[0])
        val_split = train_split + int(len(files) * splits[1])

        for i, file in enumerate(files):
            src_file = os.path.join(cls_src, file)
            if i < train_split:
                dest_file = os.path.join(cls_dest, "train", file)
            elif i < val_split:
                dest_file = os.path.join(cls_dest, "val", file)
            else:
                dest_file = os.path.join(cls_dest, "test", file)
            shutil.copy(src_file, dest_file)

split_data(src_dir, dest_dir, splits)

# Random flip, Small rotation, Slight color jitter
data_aug = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),   # ~ ±18°
    keras.layers.RandomContrast(0.1),   # slight jitter
    keras.layers.RandomBrightness(0.1)
])