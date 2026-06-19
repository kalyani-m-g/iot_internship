import streamlit as st
import pandas as pd
import joblib

model = joblib.load("iot_ids_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("IoT Traffic Classification")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=None
)

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # Remove same columns used during training
    data = data.drop(
        columns=["Unnamed: 0", "id.orig_p", "id.resp_p"],
        errors="ignore"
    )

    # Optional: show shape after preprocessing
    st.write("Dataset shape:", data.shape)

    st.write("Current columns:")
st.write(data.columns.tolist())

if hasattr(scaler, "feature_names_in_"):
    st.write("Expected columns:")
    st.write(list(scaler.feature_names_in_))

    missing = set(scaler.feature_names_in_) - set(data.columns)
    extra = set(data.columns) - set(scaler.feature_names_in_)

    st.write("Missing columns:", missing)
    st.write("Extra columns:", extra)

    data_scaled = scaler.transform(data)

    predictions = model.predict(data_scaled)

    data["Prediction"] = predictions

    st.dataframe(data)

    csv = data.to_csv(index=False)

    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv"
    )
