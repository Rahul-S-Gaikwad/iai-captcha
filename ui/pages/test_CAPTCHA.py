def render():
    import streamlit as st
    import os
    import glob
    import pandas as pd
    from object.testing.solver import Solver
    import object.config as config

    st.header("🧪 Test CAPTCHA Dataset")

    if st.button("📊 Run Test Evaluation"):
        files = sorted(glob.glob(os.path.join(config.TEST_DIR, "*.html")))
        model_path = sorted(glob.glob(os.path.join(config.MODELS_DIR, "*.keras")))[-1]
        solver = Solver(network_path=model_path)

        results = []
        for html_file in files:
            pred = solver.solve(html_file)
            with open(html_file.replace(".html", "_sol.txt"), "r") as f:
                actual = list(map(int, f.read().strip().split(",")))
            correct = set(pred) == set(actual)
            results.append({
                "File": os.path.basename(html_file),
                "Prediction": pred,
                "Actual": actual,
                "Result": "✅" if correct else "❌"
            })

        df = pd.DataFrame(results)
        accuracy = df["Result"].value_counts(normalize=True).get("✅", 0) * 100
        st.metric("Accuracy", f"{accuracy:.2f}%")
        st.dataframe(df)
