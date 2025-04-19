"""Convolutional Neural Network structure for CAPTCHA classification"""
from keras import models, layers

def conv_block(filters, kernel_size=(3, 3), input_shape=None):
    block = models.Sequential()
    if input_shape:
        block.add(layers.Conv2D(filters, kernel_size, padding='same', activation='relu', input_shape=input_shape))
    else:
        block.add(layers.Conv2D(filters, kernel_size, padding='same', activation='relu'))
    block.add(layers.BatchNormalization())
    block.add(layers.MaxPooling2D(pool_size=(2, 2)))
    return block

def get_model(n_classes):
    model = models.Sequential()

    # Input block
    model.add(conv_block(32, input_shape=(64, 64, 4)))
    model.add(conv_block(64))
    model.add(conv_block(128))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
