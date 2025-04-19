"""Train network and store trained model for CAPTCHA classification"""
import os
import pickle
import datetime
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks

import object.config as config
import object.training.network as network
import object.training.prepare_training as prepare_training


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


def train_model():
    print("📂 Loading image data...")
    X_train, X_val, Y_train, Y_val = prepare_training.training_val_data()

    with open(config.MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    model = network.get_model(len(lb.classes_))
    model.summary()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(config.MODELS_DIR, f"model_{timestamp}.keras")
    history_path = os.path.join(config.MODELS_DIR, f"history_{timestamp}.csv")

    print(f"💾 Saving model to: {model_path}")
    print(f"📝 Saving training history to: {history_path}")

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(model_path),
        verbose=2
    )

    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print("✅ Training complete.")


if __name__ == '__main__':
    train_model()
