import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from PIL import Image, ImageDraw
import tempfile
import shutil
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from object.testing.solver import Solver
import object.testing.image_extractor as image_extractor


def render():
    st.set_page_config(layout="wide")
    st.title("🔍 Solve CAPTCHA")
    st.markdown("Upload one or more `.html` CAPTCHA files, extract tiles, view predictions with confidence, and download reports.  \n💡 *Tip: You can also inspect specific tiles!*")

    uploaded_files = st.file_uploader("📄 Upload CAPTCHA HTML file(s)", type=["html", "htm"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        full_df = pd.DataFrame()
        class_metrics = {}

        for uploaded_file in uploaded_files:
            st.markdown(f"---\n### 📄 Processing: `{uploaded_file.name}`")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                html_path = tmp_file.name

            img_output_dir = html_path.replace(".html", "") + "_images"
            extracted = image_extractor.extract(html_filename=html_path, images_foldername=img_output_dir)

            if not extracted:
                st.error(f"❌ Failed to extract tiles from {uploaded_file.name}")
                continue

            solver = Solver()
            predictions, confidences, all_image_ids = solver.solve(html_path, return_confidence=True)

            actual_path = html_path.replace(".html", "_sol.txt")
            actual_ids = []
            if os.path.exists(actual_path):
                with open(actual_path, "r") as sol_file:
                    actual_ids = list(map(int, sol_file.read().strip().split(",")))

            records = []
            all_images = sorted(glob.glob(f"{img_output_dir}/**/*.png", recursive=True))
            for path in all_images:
                img_id = int(os.path.splitext(os.path.basename(path))[0])
                predicted = img_id in predictions
                actual = img_id in actual_ids
                label = "✅ MATCH" if predicted and actual else "❌ MISS"
                confidence = confidences.get(img_id, "-")
                class_name = os.path.basename(os.path.dirname(path))
                records.append({
                    "CAPTCHA": uploaded_file.name,
                    "Image ID": img_id,
                    "Class": class_name,
                    "Prediction": predicted,
                    "Actual": actual,
                    "Label": label,
                    "Confidence": round(confidence, 3) if isinstance(confidence, float) else "-",
                    "Image Path": path
                })

            df = pd.DataFrame(records).sort_values("Image ID")
            full_df = pd.concat([full_df, df], ignore_index=True)

            st.download_button(
                label="📥 Download CSV Report",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{uploaded_file.name}_predictions.csv",
                mime="text/csv"
            )

            # Original CAPTCHA layout preview (if available)
            st.subheader("🖼️ CAPTCHA Layout Preview with Predictions")
            try:
                from lxml import html as lxml_html
                with open(html_path, "r") as html_file:
                    content = html_file.read()
                tree = lxml_html.fromstring(content)
                base64_img = tree.xpath("//label[2]/@style")[0].split("url('data:image/png;base64,")[1].split("')")[0]
                from io import BytesIO
                import base64
                decoded = base64.b64decode(base64_img)
                img = Image.open(BytesIO(decoded)).convert("RGBA")
                draw = ImageDraw.Draw(img)

                for record in df[df["Prediction"]]:
                    x = (record["Image ID"] % 4) * 75 + 2
                    y = (record["Image ID"] // 4) * 75 + 2
                    draw.rectangle([x, y, x + 70, y + 70], outline="red", width=3)
                    draw.text((x + 5, y + 5), str(record["Image ID"]), fill="red")

                st.image(img, caption="Predicted IDs overlayed (approximate layout)")

            except Exception as e:
                st.warning("⚠️ Could not display CAPTCHA layout overlay.")

            if os.path.exists(img_output_dir):
                shutil.rmtree(img_output_dir)
            if os.path.exists(html_path):
                os.remove(html_path)

            # Collect per-class metrics
            if actual_ids:
                grouped = df.groupby("Class")
                for cls, group in grouped:
                    TP = sum(group["Prediction"] & group["Actual"])
                    FP = sum(group["Prediction"] & ~group["Actual"])
                    FN = sum(~group["Prediction"] & group["Actual"])
                    precision = TP / (TP + FP) if TP + FP else 0
                    recall = TP / (TP + FN) if TP + FN else 0
                    accuracy = TP / (TP + FN) if TP + FN else 0
                    class_metrics[cls] = {"Precision": precision, "Recall": recall, "Accuracy": accuracy}

        # Class Distribution
        if not full_df.empty:
            st.markdown("---")
            st.header("📊 Class Distribution of All Tiles")
            fig, ax = plt.subplots()
            full_df["Class"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Tile Count")
            ax.set_xlabel("Class")
            st.pyplot(fig)

        # Per-Class Accuracy
        if class_metrics:
            st.markdown("---")
            st.header("📉 Per-Class Accuracy Breakdown")
            metric_df = pd.DataFrame(class_metrics).T.reset_index().rename(columns={"index": "Class"})
            st.dataframe(metric_df.style.format("{:.2%}", subset=["Precision", "Recall", "Accuracy"]))

        # Tile Inspector
        st.markdown("---")
        st.subheader("🔍 Inspect a Specific Tile")
        if not full_df.empty:
            selected_id = st.number_input("Enter Image ID to inspect", min_value=0, value=0, step=1)
            matches = full_df[full_df["Image ID"] == selected_id]
            if not matches.empty:
                for _, row in matches.iterrows():
                    st.markdown(f"**CAPTCHA:** `{row['CAPTCHA']}` | **Class:** `{row['Class']}` | **Label:** `{row['Label']}` | **Conf:** `{row['Confidence']}`")
                    border_color = "green" if row["Label"] == "✅ MATCH" else "red"
                    st.markdown(f'<div style="border:3px solid {border_color}; padding:4px;">', unsafe_allow_html=True)
                    st.image(Image.open(row["Image Path"]), caption=f"ID {row['Image ID']}", use_container_width=False)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No tile found for that ID.")
