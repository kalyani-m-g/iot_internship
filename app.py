
import streamlit as st
import pandas as pd
import joblib

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="IoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #00d4ff;
}

.subtitle {
    font-size: 18px;
    color: #bdbdbd;
}

.metric-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1e1e1e;
}

.upload-box {
    border: 2px dashed #00d4ff;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("iot_ids_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.title("🛡️ IDS Dashboard")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
This project implements an **IoT Intrusion Detection System (IDS)** using the **RT-IoT2022 Dataset**.

### Features
- Detects malicious IoT traffic
- Classifies traffic category
- Identifies attack types
- Uses Machine Learning (Random Forest)
- Real-time dataset analysis

### Safe Traffic Classes
- MQTT_Publish
- Thing_Speak
- Wipro_bulb

Any other predicted class is treated as an attack.
""")

    st.markdown("---")

    st.info(
        "Upload a dataset file to analyze IoT network traffic."
    )

# ==========================
# HEADER
# ==========================

st.markdown(
    '<p class="big-title">🔍 IoT Intrusion Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Analyze IoT traffic and identify malicious activity using Machine Learning</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "📂 Upload Dataset",
    type=None
)

# ==========================
# PROCESS FILE
# ==========================

if uploaded_file is not None:

    try:

        # Try CSV first
        try:
            data = pd.read_csv(uploaded_file)
        except:
            data = pd.read_excel(uploaded_file)

        st.success("✅ Dataset uploaded successfully")

        original_shape = data.shape

        # ==========================
        # PREPROCESSING
        # ==========================

        data = data.drop(
            columns=[
                "Unnamed: 0",
                "id.orig_p",
                "id.resp_p",
                "Attack_type"
            ],
            errors="ignore"
        )

        if "proto" in data.columns or "service" in data.columns:

            data = pd.get_dummies(
                data,
                columns=[
                    col for col in ["proto", "service"]
                    if col in data.columns
                ]
            )

        expected_cols = list(scaler.feature_names_in_)

        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0

        data = data[expected_cols]

        processed_shape = data.shape

        # ==========================
        # PREDICTION
        # ==========================

        data_scaled = scaler.transform(data)

        predictions = model.predict(data_scaled)

        st.subheader("Raw Predicted Classes")
        st.write(pd.Series(predictions).value_counts())

        normal_classes = [
            "MQTT_Publish",
            "Thing_Speak",
            "Wipro_bulb"
        ]

        result_df = data.copy()

        result_df["Predicted_Class"] = predictions

        result_df["Status"] = result_df[
            "Predicted_Class"
        ].apply(
            lambda x:
            "🟢 SAFE"
            if x in normal_classes
            else "🔴 ATTACK"
        )

        # ==========================
        # SUMMARY
        # ==========================

        safe_count = (
            result_df["Status"] == "🟢 SAFE"
        ).sum()

        attack_count = (
            result_df["Status"] == "🔴 ATTACK"
        ).sum()

        total = len(result_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Records",
                f"{total:,}"
            )

        with col2:
            st.metric(
                "Safe Traffic",
                f"{safe_count:,}"
            )

        with col3:
            st.metric(
                "Attack Traffic",
                f"{attack_count:,}"
            )

        st.markdown("---")

        # ==========================
        # DATASET INFO
        # ==========================

        st.subheader("📊 Dataset Information")

        c1, c2 = st.columns(2)

        with c1:
            st.info(
                f"Original Shape: {original_shape}"
            )

        with c2:
            st.info(
                f"Processed Shape: {processed_shape}"
            )

        # ==========================
        # OVERALL RESULT
        # ==========================

        st.subheader("🛡️ Security Assessment")

        if attack_count == 0:

            st.success(
                f"""
                No attacks detected.

                All {safe_count:,} records were classified as SAFE traffic.
                """
            )

        else:

            st.error(
                f"""
                Attack traffic detected.

                {attack_count:,} out of {total:,} records were classified as malicious.
                """
            )

        # ==========================
        # ATTACK TYPES
        # ==========================

        attacks = result_df[
            result_df["Status"] == "🔴 ATTACK"
        ]

        if len(attacks) > 0:

            st.subheader("⚠️ Detected Attack Types")

            attack_counts = (
                attacks["Predicted_Class"]
                .value_counts()
            )

            st.bar_chart(attack_counts)

            st.dataframe(
                attack_counts.rename(
                    "Count"
                )
            )

        # ==========================
        # SAFE TRAFFIC TYPES
        # ==========================

        st.subheader("📈 Traffic Classification")

        st.bar_chart(
            result_df["Predicted_Class"]
            .value_counts()
        )

        # ==========================
        # SAMPLE RESULTS
        # ==========================

        st.subheader("📋 Sample Predictions")

        display_df = result_df[
            ["Predicted_Class", "Status"]
        ].copy()

        st.dataframe(
            display_df.head(100),
            use_container_width=True
        )

        # ==========================
        # DOWNLOAD
        # ==========================

        download_df = result_df[
            ["Predicted_Class", "Status"]
        ]

        csv = download_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Prediction Results",
            csv,
            "prediction_results.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(
            f"Processing Error: {str(e)}"
        )

