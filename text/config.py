import os

"""Global parameters"""
EPOCHS = 25
BATCH_SIZE = 80
SPLIT_RATE = 0.8  # 80% of our data is used as training data, 20% as validation data

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw_training_data"))
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "training_data"))
VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "validation_data"))
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_data"))

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
MODEL_LABELS_FILENAME = os.path.join(MODELS_DIR, "model_labels.dat")
