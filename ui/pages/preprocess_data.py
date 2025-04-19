def render():
    import streamlit as st
    import os
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from collections import Counter
    from object.preprocessing.create_training_data import create
    import object.config as config

    st.header("🧹 Data Preprocessing")
    st.markdown("Run the data preprocessing pipeline and inspect transformations across stages.")

    if st.button("🚀 Run Preprocessing"):
        create()
        st.success("✅ Data preprocessed successfully!")

    # Sample image previews
    st.subheader("📸 Image Previews")
    col1, col2, col3 = st.columns(3)
    def load_random_image(folder):
        if not os.path.exists(folder): return None
        classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        if not classes: return None
        selected_class = random.choice(classes)
        img_path = os.path.join(folder, selected_class, random.choice(os.listdir(os.path.join(folder, selected_class))))
        return Image.open(img_path), selected_class

    for label, folder, col in [("Raw", config.RAW_DIR, col1), ("Train", config.TRAIN_DIR, col2), ("Val", config.VAL_DIR, col3)]:
        img, cls = load_random_image(folder)
        if img:
            col.image(img, caption=f"{label} - {cls}", use_container_width=True)

    # Class distribution
    st.subheader("📊 Class Distribution")
    def class_counts(path):
        return {cls: len(os.listdir(os.path.join(path, cls))) for cls in os.listdir(path) if os.path.isdir(os.path.join(path, cls))}

    if os.path.exists(config.TRAIN_DIR):
        train_dist = class_counts(config.TRAIN_DIR)
        df = pd.DataFrame(train_dist.items(), columns=["Class", "Count"]).sort_values("Count", ascending=False)
        st.bar_chart(data=df.set_index("Class"))
        st.dataframe(df)
