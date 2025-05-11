import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('carbon_emission_predictor.sav', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("Carbon Emission Prediction App")

# User input section
st.header("Enter vehicle specifications:")

# Example inputs — adjust these to your dataset
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1)
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, step=1)
fuel_type = st.selectbox("Fuel Type", ['petrol', 'diesel', 'ethanol', 'electric'])

# Create DataFrame for prediction
input_df = pd.DataFrame({
    'Engine_Size_L': [engine_size],
    'Cylinders': [cylinders],
    'Fuel_Type': [fuel_type]
})

# Predict button
if st.button("Predict CO2 Emission"):
    prediction = model.predict(input_df)
    st.success(f"Predicted CO2 Emission: {prediction[0]:.2f} g/km")

