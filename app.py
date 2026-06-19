```python
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("iot_ids_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="IoT Traffic Classification",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 IoT Traffic Classification System")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=None
)

if uploaded_file is not None:

    try:

        # -----------------------------
        # Read file automatically
        # -----------------------------
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            data = pd.read_csv(uploaded_file)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            data = pd.read_excel(uploaded_file)

        else:
            st.error(
                "Unsupported file format. Please upload CSV or Excel files."
            )
            st.stop()

        st.success("File uploaded successfully!")

        st.write("Original Shape:", data.shape)

        # -----------------------------
        # Remove columns not used
        # -----------------------------
        data = data.drop(
            columns=[
                "Unnamed: 0",
                "id.orig_p",
                "id.resp_p",
                "Attack_type"
            ],
            errors="ignore"
        )

        # -----------------------------
        # One-hot encoding
        # -----------------------------
        if "proto" in data.columns or "service" in data.columns:

            data = pd.get_dummies(
                data,
                columns=[
                    col for col in ["proto", "service"]
                    if col in data.columns
                ]
            )

        # -----------------------------
        # Match training features
        # -----------------------------
        expected_cols = list(scaler.feature_names_in_)

        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0

        data = data[expected_cols]

        st.write("Processed Shape:", data.shape)

        # -----------------------------
        # Scale
        # -----------------------------
        data_scaled = scaler.transform(data)

        # -----------------------------
        # Predict
        # -----------------------------
        predictions = model.predict(data_scaled)

        result_df = data.copy()
        result_df["Prediction"] = predictions

        st.subheader("Prediction Results")

        st.dataframe(result_df.head(100))

        st.subheader("Prediction Counts")

        st.bar_chart(
            result_df["Prediction"].value_counts()
        )

        # -----------------------------
        # Download
        # -----------------------------
        csv = result_df.to_csv(index=False)

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")
```
