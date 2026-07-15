import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf
# ==========================================================
# Credit Card Fraud Detection System
# Multi-Task Deep Learning Streamlit Application
# Author: Amirali
# ==========================================================


# =========================
# Import Libraries
# =========================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# Custom CSS Styling
# =========================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }


    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #00c6ff;
        margin-bottom: 5px;
    }


    /* Subtitle */
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #bbbbbb;
        margin-bottom: 30px;
    }


    /* Cards */
    .custom-card {

        background-color: #161b22;

        padding: 25px;

        border-radius: 15px;

        border: 1px solid #30363d;

        margin-bottom: 20px;

    }


    /* Result title */

    .result-title {

        font-size: 28px;

        font-weight: bold;

        color: white;

    }


    /* Fraud */

    .fraud-danger {

        background-color: #3d1010;

        padding: 20px;

        border-radius: 12px;

        border-left: 6px solid #ff4b4b;

    }


    /* Safe */

    .fraud-safe {

        background-color: #103d1b;

        padding: 20px;

        border-radius: 12px;

        border-left: 6px solid #00ff88;

    }


    /* Footer */

    .footer {

        text-align:center;

        color:#777;

        margin-top:40px;

        font-size:14px;

    }


    </style>

    """,
    unsafe_allow_html=True
)



# =========================
# Load Saved Model Files
# =========================

@st.cache_resource
def load_model_files():

    """
    Load trained neural network model
    and preprocessing objects.

    cache_resource prevents reloading
    the model every time the user interacts.
    """

    try:

        model = tf.keras.models.load_model(
            "../model/fraud_ann_model.keras"
        )

        preprocessor = joblib.load(
            "../model/preprocessor.pkl"
        )

        risk_encoder = joblib.load(
            "../model/risk_encoder.pkl"
        )


        return model, preprocessor, risk_encoder


    except Exception as e:

        st.error(
            f"Model loading failed: {e}"
        )

        st.stop()



model, preprocessor, risk_encoder = load_model_files()



# =========================
# Prediction Function
# =========================

def predict_transaction(input_data):

    """
    Receive raw transaction data,
    preprocess it,
    and return model predictions.
    """


    # Apply same preprocessing pipeline
    # used during model training

    processed_data = preprocessor.transform(
        input_data
    )


    # Multi-output prediction

    prediction = model.predict(
        processed_data,
        verbose=0
    )


    fraud_prediction = prediction[0]

    risk_prediction = prediction[1]

    score_prediction = prediction[2]


    # Convert outputs

    fraud_label = int(
        fraud_prediction[0][0] > 0.5
    )


    risk_level = risk_encoder.inverse_transform(
        [
            np.argmax(
                risk_prediction[0]
            )
        ]
    )[0]


    fraud_score = float(
        score_prediction[0][0]
    )


    return (
        fraud_label,
        risk_level,
        fraud_score
    )



# =========================
# Sidebar
# =========================

with st.sidebar:


    st.image(
        "../Streamlit.png",
        use_container_width=True
    )


    st.title(
        "💳 Fraud Detection AI"
    )


    st.write(
        """
        This application uses a
        Multi-Task Deep Neural Network
        to analyze credit card transactions.
        """
    )


    st.divider()


    st.subheader(
        "Model Outputs"
    )


    st.write(
        """
        🔹 Fraud Detection

        🔹 Risk Level Classification

        🔹 Fraud Probability Score
        """
    )


    st.divider()


    st.caption(
        "Built with TensorFlow, "
        "Scikit-Learn and Streamlit"
    )



# =========================
# Main Header
# =========================


st.markdown(
    """
    <div class="main-title">

    💳 Credit Card Fraud Detection System

    </div>


    <div class="sub-title">

    Deep Learning • Multi-Task Learning • TensorFlow

    </div>

    """,

    unsafe_allow_html=True
)



st.markdown(
    """
    <div class="custom-card">

    This AI system predicts:

    <br>

    🚨 Whether a transaction is fraudulent

    <br>

    📊 Fraud risk category

    <br>

    🎯 Fraud probability score

    </div>

    """,

    unsafe_allow_html=True
)
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




# ==========================================================
# Transaction Input Section
# ==========================================================


st.markdown(
    "## 📝 Enter Transaction Information"
)


# Creating two columns
# to make the form cleaner

col1, col2 = st.columns(2)



with col1:


    transaction_amount = st.number_input(
        "💵 Transaction Amount",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


    transaction_hour = st.slider(
        "🕒 Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )


    merchant_category = st.selectbox(
        "🏪 Merchant Category",
        [
            "grocery",
            "electricity",
            "travel"
        ]
    )


    distance_from_home = st.number_input(
        "📍 Distance From Home (km)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )



with col2:


    device_type = st.selectbox(
        "📱 Device Type",
        [
            "laptop",
            "mobile"
        ]
    )


    previous_flag = st.selectbox(
        "⚠️ Previous Fraud History",
        [
            "yes",
            "no"
        ]
    )


    account_age = st.slider(
        "🏦 Account Age (Years)",
        min_value=3,
        max_value=11,
        value=5
    )


    st.write("")



# ==========================================================
# Prediction Button
# ==========================================================


predict_button = st.button(
    "🚀 Analyze Transaction",
    use_container_width=True
)



if predict_button:


    # Creating dataframe
    # with exactly the same features
    # used during training

    input_data = pd.DataFrame(
        [
            {

                "transaction_amount":
                    transaction_amount,


                "transaction_hour":
                    transaction_hour,


                "merchant_category":
                    merchant_category,


                "distance_from_home_km":
                    distance_from_home,


                "device_type":
                    device_type,


                "previous_flag":
                    previous_flag,


                "account_age_years":
                    account_age

            }
        ]
    )



    # Showing loading animation

    with st.spinner(
        "🤖 AI model is analyzing transaction..."
    ):


        fraud_label, risk_level, fraud_score = (
            predict_transaction(
                input_data
            )
        )



    st.divider()



    # ======================================================
    # Prediction Results
    # ======================================================


    st.markdown(
        "## 📊 Prediction Results"
    )



    result_col1, result_col2, result_col3 = st.columns(3)



    # -------------------------
    # Fraud Status
    # -------------------------

    with result_col1:


        if fraud_label == 1:


            st.markdown(
                """
                <div class="fraud-danger">

                <h3>🚨 Fraud Detected</h3>

                <p>
                This transaction looks suspicious.
                </p>

                </div>
                """,

                unsafe_allow_html=True
            )


        else:


            st.markdown(
                """
                <div class="fraud-safe">

                <h3>✅ Safe Transaction</h3>

                <p>
                No fraud pattern detected.
                </p>

                </div>
                """,

                unsafe_allow_html=True
            )



    # -------------------------
    # Risk Level
    # -------------------------

    with result_col2:


        st.metric(
            label="Risk Level",
            value=risk_level.upper()
        )



    # -------------------------
    # Fraud Score
    # -------------------------

    with result_col3:


        st.metric(
            label="Fraud Score",
            value=f"{fraud_score:.2f}"
        )



    st.write("")



    # ======================================================
    # Fraud Probability Visualization
    # ======================================================


    st.markdown(
        "### 🎯 Fraud Probability Meter"
    )



    # Convert score to percentage

    probability = min(
        max(fraud_score, 0),
        100
    )



    st.progress(
        int(probability)
    )



    st.write(
        f"Model confidence score: **{probability:.2f}%**"
    )



    # ======================================================
    # Detailed Explanation
    # ======================================================


    with st.expander(
        "ℹ️ How this prediction works"
    ):


        st.write(
            """
            The model uses a Multi-Task Neural Network.

            The network simultaneously performs:

            🔹 Binary classification:
            Fraud / Normal transaction

            🔹 Multi-class classification:
            Low / Medium / High risk

            🔹 Regression:
            Fraud probability score estimation

            The same preprocessing pipeline
            used during training is applied here.
            """
        )



# ==========================================================
# Footer
# ==========================================================


st.markdown(
    """
    <div class="footer">

    Developed by Amirali |
    TensorFlow + Scikit-Learn + Streamlit

    </div>

    """,

    unsafe_allow_html=True
)

