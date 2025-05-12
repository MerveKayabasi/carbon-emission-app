import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop, errors='ignore')


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
transmission = st.selectbox("Transmission Type", ['automatic', 'manual', 'other'])
vehicle_type = st.selectbox("Vehicle Type", ['SUV', 'sedan', 'pickup', 'other'])

fuel_comb_mpg = st.number_input("Fuel Consumption (Comb MPG)", min_value=0.0, max_value=100.0, step=0.1)
fuel_comb_l = st.number_input("Fuel Consumption (Comb L/100km)", min_value=0.0, max_value=30.0, step=0.1)

# Create DataFrame for prediction
input_df = pd.DataFrame({
    'Engine_Size_L': [engine_size],
    'Cylinders': [cylinders],
    'Fuel_Type': [fuel_type],
     'Transmission': [transmission],
    'vehicle_type': [vehicle_type],
    'Fuel_Consumption_Comb_mpg': [fuel_comb_mpg],
    'Fuel_Consumption_Comb_L_per_100_km': [fuel_comb_l]
})

# Predict button
if st.button("Predict CO2 Emission"):
    prediction = model.predict(input_df)
    st.success(f"Predicted CO2 Emission: {prediction[0]:.2f} g/km")