import streamlit as st
import pandas as pd
import joblib

model = joblib.load("iot_ids_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("IoT Traffic Classification")

uploaded_file = st.file_uploader(
    "Upload CSV with 90 features",
    type=["csv"]
)

if uploaded_file:

    data = pd.read_csv(uploaded_file)

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
