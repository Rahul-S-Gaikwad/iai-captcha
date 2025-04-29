"""Train network and store trained model for chosen subfolder"""
import os
import pickle
import datetime
import pandas as pd
from keras import callbacks

import network
import merged.config as config
import prepare_training


def train_model(captcha_type):
    USE_GPU = False     # Change to True when using GPU-KONG
    if USE_GPU:
        GPU_COUNT = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_COUNT)

    # Get training and validation data from specified subfolder
    print("📂 Loading image data...")
    X_train, X_val, Y_train, Y_val = prepare_training.training_val_data(captcha_type)

    with open(os.path.join(config.MODELS_DIR, captcha_type, config.MODEL_LABELS_FILENAME), "rb") as f:
        lb = pickle.load(f)

    model = network.get_model(len(lb.classes_))

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(config.MODELS_DIR, captcha_type, f"model_{timestamp}.keras")
    history_path = os.path.join(config.MODELS_DIR, captcha_type, f"history_{timestamp}.csv")

    print(f"💾 Saving model to: {model_path}")
    print(f"📝 Saving training history to: {history_path}")

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=get_callbacks(model_path))

    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print("✅ Training complete.")


def get_callbacks(model_save_path):
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
        )
    ]

if __name__ == '__main__':
    train_model("object")

