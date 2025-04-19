import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import object.config as config
import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import glob
import io
from contextlib import redirect_stdout

st.title("🧬 Model Details & Insights")

# Latest model
model_files = sorted(glob.glob(os.path.join(config.MODELS_DIR, "*.keras")))
if not model_files:
    st.warning("No trained models found.")
    st.stop()

latest_model_path = model_files[-1]
model = keras.models.load_model(latest_model_path)

st.success(f"✅ Loaded model: `{os.path.basename(latest_model_path)}`")

# Summary
st.subheader("📄 Model Summary")
with st.expander("🔍 Expand model.summary()"):
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.summary()
    st.code(buf.getvalue())

# Plot
st.subheader("🧬 Architecture Diagram")
try:
    plot_path = os.path.join(config.MODELS_DIR, "model_plot.png")
    plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
    st.image(plot_path)
except:
    st.warning("⚠️ Could not render diagram. Install `pydot` and `graphviz`.")

# Config
st.subheader("⚙️ Configuration & Hyperparameters")
col1, col2 = st.columns(2)
with col1:
    st.write("**Epochs:**", config.EPOCHS)
    st.write("**Batch Size:**", config.BATCH_SIZE)
with col2:
    st.write("**Loss Function:**", model.loss)
    st.write("**Optimizer:**", model.optimizer.__class__.__name__)
    st.write("**Params:**", f"{model.count_params():,}")

# Download model
with st.expander("📥 Download Model"):
    with open(latest_model_path, "rb") as f:
        st.download_button("⬇️ Download Model (.keras)", f, file_name=os.path.basename(latest_model_path))
