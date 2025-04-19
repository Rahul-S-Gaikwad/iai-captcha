import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# ✅ Set layout early (MUST be first Streamlit call)
st.set_page_config(page_title="IAI CAPTCHA Solver", layout="wide")

st.title("🤖 IAI CAPTCHA Solver Dashboard")
st.markdown("""
Welcome to the **IAI CAPTCHA Solver**. This tool lets you preprocess image data, train machine learning models, test and validate CAPTCHA solving performance, and explore model details.

### 🔧 Available Operations
- **🧹 Data Preprocessing**: Transform raw data into training/validation sets and explore class distribution.
- **🧠 Train Model**: Train a CNN to classify CAPTCHA tiles.
- **🔍 Solve CAPTCHA**: Upload HTML CAPTCHA files and get predictions with visuals.
- **📊 Compare Models**: Compare trained models on accuracy, confusion matrices, and per-class scores.
- **🧬 Model Details**: View model architecture, training configuration, and metadata.
""")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "🧹 Preprocess Data",
        "🧠 Train Model",
        "🔍 Solve CAPTCHA",
        "📊 Compare Models",
        "🧬 Model Details"
    ]
)

if page == "🧹 Preprocess Data":
    from ui.pages import preprocess_data
    preprocess_data.render()

elif page == "🧠 Train Model":
    from ui.pages import train_model
    train_model.render()

elif page == "🔍 Solve CAPTCHA":
    from ui.pages import solve_captcha
    solve_captcha.render()

elif page == "📊 Compare Models":
    from ui.pages import compare_models
    compare_models.render()

elif page == "🧬 Model Details":
    from ui.pages import model_info
    model_info.render()