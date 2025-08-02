import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pickle

st.title("📊 Customer Churn Prediction App")

# Load model and column list
model, model_columns = pickle.load(open("xgb_model.pkl", "rb"))

# UI Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 24)
monthly_charges = st.slider("Monthly Charges", 10.0, 150.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Prepare input data
input_data = {
    'SeniorCitizen': senior_citizen,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == "Male" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0
}

X_input = pd.DataFrame([input_data])

# Align with training columns
for col in model_columns:
    if col not in X_input.columns:
        X_input[col] = 0

X_input = X_input[model_columns]

# Predict
if st.button("Predict"):
    pred_prob = model.predict_proba(X_input)[0][1]
    st.write(f"💡 Churn Probability: **{pred_prob:.2f}**")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_input)

    st.subheader("🔍 SHAP Explanation")

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
