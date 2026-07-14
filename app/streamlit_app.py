import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

import numpy as np

# Load model and preprocessing
model = tf.keras.models.load_model("../model/fraud_ann_model.keras")
preprocessor = joblib.load("../model/preprocessor.pkl")
risk_encoder = joblib.load("../model/risk_encoder.pkl")

st.title("💳 Credit Card Fraud Detection System")

st.header("Enter Transaction Details")

transaction_amount = st.number_input("Transaction Amount", value=10.0)
transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
merchant_category = st.selectbox(
    "Merchant Category", ["grocery", "electricity", "travel"]
)
distance_from_home = st.number_input("Distance From Home (km)", value=5.0)
device_type = st.selectbox("Device Type", ["laptop", "mobile"])
previous_flag = st.selectbox("Previous Fraud Flag", ["yes", "no"])
account_age = st.slider("Account Age (years)", 3, 11, 5)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [
            {
                "transaction_amount": transaction_amount,
                "transaction_hour": transaction_hour,
                "merchant_category": merchant_category,
                "distance_from_home_km": distance_from_home,
                "device_type": device_type,
                "previous_flag": previous_flag,
                "account_age_years": account_age,
            }
        ]
    )

    processed = preprocessor.transform(input_data)

    pred_label, pred_risk, pred_score = model.predict(processed)

    fraud_class = int(pred_label[0][0] > 0.5)
    risk_class = risk_encoder.inverse_transform([pred_risk.argmax()])[0]
    fraud_score = float(pred_score[0][0])

    st.subheader("Prediction Results")

    st.write("Fraud Label:", fraud_class)
    st.write("Risk Level:", risk_class)
    st.write("Fraud Probability Score:", round(fraud_score, 2))
