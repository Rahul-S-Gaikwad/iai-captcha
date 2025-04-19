"""Preparing training & validation data for CAPTCHA classifier"""
import os
import numpy as np
from PIL import Image
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import pickle
import object.config as config
from collections import Counter


def load_images_and_labels(data_dir):
    X, Y = [], []

    for image_file in paths.list_images(data_dir):
        try:
            image = Image.open(image_file).convert("RGBA").resize((64, 64))
            label = os.path.basename(os.path.dirname(image_file))
            X.append(np.array(image, dtype=np.float32) / 255)
            Y.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {image_file}: {e}")

    return np.array(X), np.array(Y)


def training_val_data():
    X_train, Y_train = load_images_and_labels(config.TRAIN_DIR)
    X_val, Y_val = load_images_and_labels(config.VAL_DIR)

    print(f"📊 Class distribution (train): {Counter(Y_train)}")

    lb = LabelBinarizer()
    Y_train_enc = lb.fit_transform(Y_train)
    Y_val_enc = lb.transform(Y_val)

    with open(config.MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    return X_train, X_val, Y_train_enc, Y_val_enc
