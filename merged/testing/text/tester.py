"""Measure test accuracy of models for TRZ"""
import os
import glob

from solver import Solver
import merged.config as config


DEBUG = False       # if True script will output predictions compared to solutions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow error messages

def get_latest_model_path():
    return sorted(glob.glob(os.path.join(config.MODELS_DIR, "text", "model_*.keras")))[-1]


MODEL_PATH = get_latest_model_path()
# MODEL_PATH = os.path.join(config.MODELS_DIR, "text", f"model_20250421_002546.keras")

def test():
    print(f"\nMarketplace: TRZ; Network: {MODEL_PATH}\n")

    total_err = 0
    total = 0

    # List of raw test captchas
    captcha_image_files = glob.glob(os.path.join(config.TEST_DIR, "text", "*"))

    solver = Solver(network_path=MODEL_PATH)
    for img_file in captcha_image_files:
        # Get predicted labels of captcha
        prediction = solver.solve(img_file)

        # Get actual labels of captcha
        filename = os.path.basename(img_file)
        captcha_text = os.path.splitext(filename)[0]
        correct_labels = [char for char in captcha_text]

        # Track errors
        if prediction != correct_labels:
            total_err += 1
            print(f"Predicted: {prediction} Correct: {correct_labels}")
        total += 1

        # Debug mode for more information
        if DEBUG:
            print(f"Predicted: {prediction} Correct: {correct_labels}")

    print(f"Solved {total-total_err} captchas of {total} correctly. Accuracy = {(total-total_err)/total}")



if __name__ == '__main__':
    test()

