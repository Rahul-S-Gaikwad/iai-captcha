"""Measure test accuracy of models for WHM"""
import glob
import os

from solver import Solver
import merged.config as config

DEBUG = False       # if True script will output predictions compared to solutions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow error messages

def get_latest_model_path():
    return sorted(glob.glob(os.path.join(config.MODELS_DIR, "object", "model_*.keras")))[-1]


MODEL_PATH = get_latest_model_path()
# MODEL_PATH = os.path.join(config.MODELS_DIR, "object", f"model_20250324_180838.keras")

def test():
    print(f"\nMarketplace: WHM; Network: {MODEL_PATH}\n")

    total_err = 0
    total = 0

    # List of raw test captchas
    captcha_image_files = glob.glob(os.path.join(config.TEST_DIR, "object", "*.html"))

    solver = Solver(network_path=MODEL_PATH)
    for captcha in captcha_image_files:
        # Get predicted ids of captcha
        prediction = solver.solve(captcha)

        # Get actual ids of captcha
        with open(captcha.replace(".html", "_sol.txt")) as solution:
            sol_string = solution.readline().strip()
            correct_ids = sol_string.split(",")
            ids = [int(img_id) for img_id in correct_ids]
            ids.sort()

            # Track errors
            if prediction != ids:
                total_err += 1
                print(f"Predicted: {prediction} Correct: {ids}")
            total += 1

        # Debug mode for more information
        if DEBUG:
            print(f"Current filename: {captcha}")
            print(f"Predicted: {prediction} Correct: {ids}")

    print(f"Solved {total-total_err} captchas of {total} correctly. Accuracy = {(total-total_err)/total}")


if __name__ == '__main__':
    test()
