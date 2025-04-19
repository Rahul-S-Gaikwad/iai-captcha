"""CAPTCHA solver for WHM that can be used by passing the path to the HTML as argument"""
import tensorflow as tf
import glob
import os
from PIL import Image
import numpy as np
import shutil
import sys
import json
import pickle
import keras

import object.testing.image_extractor as image_extractor
import object.config as config


def get_latest_model_path():
    return sorted(glob.glob(os.path.join(config.MODELS_DIR, "model_*.keras")))[-1]


MODEL_PATH = get_latest_model_path()


class Solver:
    def __init__(self, network_path=MODEL_PATH, encoding_path=config.MODEL_LABELS_FILENAME):
        self.network_path = network_path
        self.loaded_model = keras.models.load_model(self.network_path)

        with open(encoding_path, "rb") as f:
            self.encoder = pickle.load(f)

    def solve(self, filename, return_confidence=False):
        img_foldername = filename.replace(".html", "") + "_images"
        success = image_extractor.extract(html_filename=filename, images_foldername=img_foldername)
        result = []
        confidences = {}
        image_ids = []

        if success:
            image_list = []
            captcha_ids = []
            for image in glob.glob(f"{img_foldername}{os.sep}*{os.sep}*.png"):
                label = image.split(os.sep)[img_foldername.count(os.sep) + 1]
                captcha_ids.append(
                    image.split(os.sep)[img_foldername.count(os.sep) + 2].replace(".png", "")
                )
                img = Image.open(image)
                img.load()
                image_list.append(np.asarray(img.convert("RGBA"), dtype=np.float32) / 255)

            predictions_raw = self.loaded_model.predict(np.asarray(image_list))
            predictions = self.encoder.inverse_transform(predictions_raw)
            result = [int(captcha_ids[i]) for i in range(len(predictions)) if predictions[i] == label]
            result.sort()

            for i, pid in enumerate(captcha_ids):
                confidences[int(pid)] = float(np.max(predictions_raw[i]))

            shutil.rmtree(img_foldername)

        if return_confidence:
            return result, confidences, captcha_ids
        return result


def main():
    if len(sys.argv) != 2:
        raise ValueError("Please provide the pathname to the HTML-file as an argument!")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    solver = Solver()
    result = solver.solve(filename=sys.argv[1])
    formatted_result = json.dumps(result)
    print(formatted_result)


if __name__ == "__main__":
    main()
