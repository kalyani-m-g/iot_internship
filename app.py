
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

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#f5f7fb;
}

.main{
    padding-top:0rem;
}

/* Only File Uploader label */
[data-testid="stFileUploader"] > label {
    color: black !important;
}

.hero-box{
    background: linear-gradient(135deg,#1e3a8a,#2563eb);
    padding:30px;
    border-radius:18px;
    color:white;
    margin-bottom:20px;
}

.hero-title{
    font-size:38px;
    font-weight:700;
}

.hero-sub{
    font-size:17px;
    opacity:0.9;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
}

.sidebar-title{
    color:#2563eb;
    font-weight:bold;
}

.footer{
    text-align:center;
    padding:20px;
    color:#666;
    font-size:14px;
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

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2092/2092663.png",
        width=80
    )

    st.title("IDS Dashboard")

    st.markdown("---")

    with st.expander("📌 Project Overview", expanded=True):

        st.write("""
This project detects malicious IoT network traffic using a
Random Forest based Intrusion Detection System trained on the
RT-IoT2022 dataset.

The system classifies incoming traffic as:

• SAFE Traffic

• ATTACK Traffic

and identifies the specific attack category.
""")

    with st.expander("⚙️ Key Features"):

        st.write("""
✅ Real-Time Traffic Analysis

✅ Attack Detection

✅ Traffic Classification

✅ Random Forest Model

✅ Result Export Support
""")

    with st.expander("🟢 Safe Traffic Classes"):

        st.write("""
• MQTT_Publish

• Thing_Speak

• Wipro_bulb
""")

    with st.expander("📚 Project Details"):

        st.write("""
Dataset : RT-IoT2022

Records : 123,117

Features : 82+

Classes : 12

Model : Random Forest

Task :
Intrusion Detection and Attack Classification
""")

    with st.expander("Developed by:"):
          st.write("""
KALYANI M G 


""")
   

    st.markdown("---")

    st.info(
        "Upload a dataset file and generate security predictions instantly."
    )

# ==========================
# HEADER
# ==========================

st.markdown("""
<div class="hero-box">

<div class="hero-title">
🛡️ IoT Intrusion Detection System
</div>

<div class="hero-sub">
Real-Time Analysis and Detection of Malicious IoT Network Traffic using Machine Learning
</div>

</div>
""", unsafe_allow_html=True)

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

        st.subheader("Model Classes")
        st.write(model.classes_)

        st.subheader("Prediction Distribution")
        st.write(pd.Series(predictions).value_counts())

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
            c1,c2,c3 = st.columns(3)

        with c1:
            st.info(f"📄 Total Records\n\n{total:,}")

        with c2:
            st.success(f"🟢 Safe Traffic\n\n{safe_count:,}")

        with c3:
            st.error(f"🔴 Attack Traffic\n\n{attack_count:,}")
        

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


    st.markdown("""
    <hr>

    <div class="footer">

    Developed by <b>Kalyani M G</b>

    <br>

    IoT Intrusion Detection System using Machine Learning

    </div>
    """, unsafe_allow_html=True)  
