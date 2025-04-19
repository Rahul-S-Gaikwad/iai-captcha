import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from object.testing.solver import Solver
import object.config as config

st.title("📊 Compare Trained Models")

model_files = sorted(glob.glob(os.path.join(config.MODELS_DIR, "*.keras")))
if not model_files:
    st.warning("No models found.")
    st.stop()

selected_models = st.multiselect("Select models", model_files, default=model_files[:2])

if selected_models:
    for path in selected_models:
        name = os.path.basename(path)
        st.subheader(f"🧠 Model: `{name}`")

        solver = Solver(network_path=path)

        y_true, y_pred = [], []
        for html in glob.glob(f"{config.TEST_DIR}/*.html"):
            sol_file = html.replace(".html", "_sol.txt")
            if not os.path.exists(sol_file):
                continue
            pred = solver.solve(html)
            with open(sol_file) as f:
                true = list(map(int, f.read().strip().split(",")))
            y_true.extend(true)
            y_pred.extend(pred)

        acc = sum([1 for x, y in zip(y_pred, y_true) if x == y]) / len(y_true)
        st.metric("Accuracy", f"{acc:.2%}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        st.pyplot(fig)

        # Accuracy over epochs
        history_path = path.replace("model_", "history_").replace(".keras", ".csv")
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)
            fig, ax = plt.subplots()
            ax.plot(df["accuracy"], label="Train")
            ax.plot(df["val_accuracy"], label="Validation")
            ax.set_title("Accuracy per Epoch")
            ax.set_xlabel("Epoch")
            ax.legend()
            st.pyplot(fig)
