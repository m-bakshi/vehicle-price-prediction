import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("vehicle_price_model.pkl")
encoder = joblib.load("onehotencoder.pkl")

# List of categorical and numerical features (update as per your preprocessing)
categorical_cols = [
    'make', 'model', 'engine', 'cylinders', 'fuel',
    'transmission', 'trim', 'body', 'doors',
    'exterior_color', 'interior_color', 'drivetrain'
]
numerical_cols = ['mileage', 'age']  # Add other numerical features if used

st.title("Vehicle Price Prediction App")

# --- User Input Section ---
st.header("Enter Vehicle Details")

# Collect user input for each feature
user_input = {}

for col in categorical_cols:
    user_input[col] = st.text_input(f"{col.capitalize()}")

for col in numerical_cols:
    user_input[col] = st.number_input(f"{col.capitalize()}", value=0.0)

# When user clicks Predict
if st.button("Predict Price"):
    # Prepare input data
    input_df = pd.DataFrame([user_input])

    # Ensure correct types
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
    for col in numerical_cols:
        input_df[col] = pd.to_numeric(input_df[col])

    # One-hot encode categorical features
    input_cat = encoder.transform(input_df[categorical_cols])
    input_cat_df = pd.DataFrame(
        input_cat,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Combine numerical and encoded categorical features
    input_final = pd.concat([input_df[numerical_cols], input_cat_df], axis=1)

    # Clean column names (as done during training)
    input_final.columns = input_final.columns.str.replace('[\[\]<>\s]', '_', regex=True)

    # Ensure columns match the model's expected input
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in input_final.columns:
            input_final[col] = 0  # Add missing columns as zeros
    input_final = input_final[model_features]

    # Make prediction (log scale)
    log_pred = model.predict(input_final)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"Predicted Vehicle Price: ${price_pred:,.2f}")
