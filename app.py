import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

artifact = joblib.load("customer_churn_model.joblib")

preprocessor = artifact["preprocessor"]
model = artifact["model"]
threshold = artifact["threshold"]
numerical_columns = artifact["numerical_columns"]
categorical_columns = artifact["categorical_columns"]

st.title("Customer Churn Prediction")
st.write("Predict customer churn risk using machine learning.")

st.success("Model loaded successfully.")
st.caption(f"Classification threshold: {threshold}")

st.header("Customer Details")

with st.form("customer_details_form"):
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )

    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1]
    )

    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )
    partner = st.selectbox(
        "Partner", ["Yes", "No"]
    )
    dependents = st.selectbox(
        "Dependents", ["Yes", "No"]
    )
    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    submitted = st.form_submit_button(
        "Predict Churn Risk"
    )


if submitted:
    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    customer_data = customer_data[
        numerical_columns + categorical_columns
    ]

    customer_processed = preprocessor.transform(customer_data)

    churn_probability = model.predict_proba(
        customer_processed
    )[0, 1]

    churn_prediction = int(churn_probability >= threshold)

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{churn_probability:.2%}")

    if churn_prediction == 1:
        st.error("High Risk: Customer is predicted to churn.")
    else:
        st.success("Lower Risk: Customer is predicted not to churn.")

    st.caption(
        f"Classification threshold: {threshold:.0%}. "
        "Predictions are estimates, not guarantees."
    )


