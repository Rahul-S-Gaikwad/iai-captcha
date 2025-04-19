def render():
    import streamlit as st
    import os
    import time
    import pandas as pd
    from glob import glob
    import object.config as config

    st.header("🧠 Model Training")

    st.markdown("Train the CAPTCHA classifier model and visualize training metrics.")

    if st.button("🚀 Start Training"):
        with st.spinner("Training..."):
            os.system("python object/training/trainer.py")
            st.success("✅ Training Complete!")

    history_files = sorted(glob(os.path.join(config.MODELS_DIR, "history_*.csv")))
    if not history_files:
        st.info("No training history available.")
        return

    latest = history_files[-1]
    df = pd.read_csv(latest)

    st.subheader("📈 Accuracy & Loss")
    st.line_chart(df[["accuracy", "val_accuracy"]])
    st.line_chart(df[["loss", "val_loss"]])

    if "learning_rate" in df.columns or "lr" in df.columns:
        st.line_chart(df[["learning_rate"]] if "learning_rate" in df.columns else df[["lr"]])

    with st.expander("🔢 Full Metrics Table"):
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
