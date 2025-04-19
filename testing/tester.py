"""Measure test accuracy of models for WHM with visual feedback and auto model selection."""

import glob
import os
import object.config as config
from object.testing.solver import Solver

DEBUG = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_latest_model_path():
    return sorted(glob.glob(os.path.join(config.MODELS_DIR, "model_*.keras")))[-1]


MODEL_PATH = get_latest_model_path()


def test():
    print(f"\nMarketplace: WHM; Network: {MODEL_PATH}\n")

    total_err = 0
    total = 0

    captcha_image_files = glob.glob(f"{config.TEST_DIR}{os.sep}*.html")
    solver = Solver(network_path=MODEL_PATH)

    for captcha in captcha_image_files:
        prediction = solver.solve(captcha)

        with open(captcha.replace(".html", "_sol.txt")) as solution:
            sol_string = solution.readline().strip()
            correct_ids = list(map(int, sol_string.split(",")))
            correct_ids.sort()

            if prediction != correct_ids:
                total_err += 1
                print(f"❌ {os.path.basename(captcha)}")
                print(f"   ➤ Predicted: {prediction}")
                print(f"   ➤ Correct:   {correct_ids}")
            else:
                print(f"✅ {os.path.basename(captcha)}")

            total += 1

        if DEBUG:
            print(f"[DEBUG] {captcha}")
            print(f"Prediction: {prediction} | Truth: {correct_ids}")

    print(f"\n📊 Solved {total - total_err} captchas of {total} correctly.")
    print(f"🎯 Accuracy = {(total - total_err) / total:.2%}")


if __name__ == '__main__':
    test()
