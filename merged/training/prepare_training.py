"""Preparing training & validation data for WHM network training"""
import os
import numpy as np
from PIL import Image
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import cv2
import shutil

import merged.config as config
import merged.preprocessing.text.image_splitter as image_splitter


def training_val_data(captcha_type):

    if captcha_type == "object":
        X_train, X_val, Y_train, Y_val = object_split_data()
    elif captcha_type == "text":
        X_train, X_val, Y_train, Y_val = text_split_data()
    else:
        raise ValueError("Please provide the string to a valid CAPTCHA type!")

    # Normalize data
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_val = lb.transform(Y_val)

    # Save the mapping from labels to one-hot encodings.
    lb_path = os.path.join(config.MODELS_DIR, captcha_type)
    if os.path.exists(lb_path):
        shutil.rmtree(lb_path)
    os.makedirs(lb_path, exist_ok=True)
    with open(os.path.join(lb_path, config.MODEL_LABELS_FILENAME), "wb") as f:
        pickle.dump(lb, f)

    return X_train, X_val, Y_train, Y_val


def object_split_data():
    # Loading the training dataset
    X_train = []
    Y_train = []
    # Iterate over training images
    for image_file in paths.list_images(os.path.join(config.TRAIN_DIR, "object")):
        # Load image
        image = Image.open(image_file)
        image.load()

        # Label based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add image and label to training data
        X_train.append(image)
        Y_train.append(label)

    # Permute the list, so that we don't introduce a bias by having the rotated version of a pic in a row
    sequence = np.random.permutation(len(X_train))
    X_train = [X_train[index] for index in sequence]
    X_train = np.asarray(X_train)
    Y_train = [Y_train[index] for index in sequence]
    Y_train = np.asarray(Y_train)

    # Loading the validation dataset
    X_val = []
    Y_val = []
    # Iterate over validation images
    for image_file in paths.list_images(os.path.join(config.VAL_DIR, "object")):
        # Load image
        image = Image.open(image_file)
        image.load()

        # Label based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add image and label to training data
        X_val.append(image)
        Y_val.append(label)

    X_val = np.asarray(X_val)
    Y_val = np.asarray(Y_val)


    return X_train, X_val, Y_train, Y_val



def text_split_data():
    data = []
    labels = []

    # Iterate over training images
    for image_file in paths.list_images(os.path.join(config.TRAIN_DIR, "text")):
        # Load image and convert to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize letter to 35x35 (wxh) pixels
        image = image_splitter.resize_to_fit(image, 35, 35)

        # Add third channel dimension for Keras
        image = np.expand_dims(image, axis=2)

        # Label letter based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add letter image and label to training data
        data.append(image)
        labels.append(label)

    # Divide by 255 to normalize to range 0 to 1 (pixel values range from 0 to 256)
    data = np.array(data)
    labels = np.array(labels)

    # Split into training and validation data
    (X_train, X_val, Y_train, Y_val) = train_test_split(data, labels, train_size=config.SPLIT_RATE, random_state=0)

    return X_train, X_val, Y_train, Y_val





