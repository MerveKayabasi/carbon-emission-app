import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# ----- Custom Transformer (if used during training) -----
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop, errors='ignore')

# ----- Load the Trained Model -----
with open('carbon_emission_predictor.sav', 'rb') as file:
    model = pickle.load(file)

# ----- Page Settings -----
st.set_page_config(page_title="CO2 Emission Predictor", layout="centered")
st.title("🚗 CO₂ Emission Prediction App")
st.markdown("This application predicts the estimated **carbon dioxide emissions (g/km)** for a vehicle based on its specifications.")
st.caption("📌 Note: Predictions are based on historical training data and may vary from real-world values.")

st.subheader("Enter Vehicle Specifications")

# ----- Top Brands & Models -----
top_brands = ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'BMW', 'Hyundai', 'Volkswagen', 'Mazda', 'Mercedes-Benz']
top_models = ['f-150_ffv', 'f-150_ffv_4x4', 'mustang', 'focus_ffv', 'f-150_4x4', 'f-150', 'sonic_5', 'ats', 'jetta', 'compass']

# ----- Input Fields -----
col1, col2 = st.columns(2)
with col1:
    brand_input = st.selectbox("Vehicle Brand", ["Select..."] + sorted(top_brands + ['Other']))
with col2:
    model_input = st.selectbox("Vehicle Model", ["Select..."] + sorted(top_models + ['Other']))

col3, col4 = st.columns(2)
with col3:
    engine_size = st.number_input("Engine Size (Liters)", min_value=0.0, max_value=8.4, step=0.1)
with col4:
    cylinders = st.selectbox("Number of Cylinders", ["Select...", 3, 4, 5, 6, 8, 10, 12, 16])

col5, col6 = st.columns(2)
with col5:
    fuel_type = st.selectbox("Fuel Type", ["Select...", 'petrol', 'diesel', 'LPG', 'ethanol', 'natural_gas'])
with col6:
    transmission = st.selectbox("Transmission Type", ["Select...",
        'AS5', 'M6', 'AV7', 'AS6', 'AM6', 'A6', 'AM7', 'AV8', 'AS8',
        'A7', 'A8', 'M7', 'A4', 'M5', 'AV', 'A5', 'AS7', 'A9', 'AS9',
        'AV6', 'AS4', 'AM5', 'AM8', 'AM9', 'AS10', 'A10', 'AV10'
    ])

col7, col8 = st.columns(2)
with col7:
    vehicle_type = st.selectbox("Vehicle Type", ["Select...",
        'compact_car', 'small_suv', 'medium_car', 'two_seater_sport', 'very_small_car',
        'subcompact', 'large_car', 'small_station_wagon', 'standard_suv', 'cargo_van',
        'passenger_van', 'large_pickup_truck', 'minivan', 'special_vehicle',
        'medium_station_wagon', 'small_pickup_truck'
    ])
with col8:
    fuel_comb_mpg = st.number_input("Fuel Consumption (MPG)", min_value=0.0, max_value=69.0, step=1.0)

fuel_comb_l = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=26.1, step=0.1)

# ----- Validation -----
missing_fields = []
if brand_input == "Select...": missing_fields.append("Vehicle Brand")
if model_input == "Select...": missing_fields.append("Vehicle Model")
if fuel_type == "Select...": missing_fields.append("Fuel Type")
if transmission == "Select...": missing_fields.append("Transmission Type")
if vehicle_type == "Select...": missing_fields.append("Vehicle Type")
if cylinders == "Select...": missing_fields.append("Number of Cylinders")
if engine_size == 0.0: missing_fields.append("Engine Size")
if fuel_comb_mpg == 0.0: missing_fields.append("Fuel Consumption (MPG)")
if fuel_comb_l == 0.0: missing_fields.append("Fuel Consumption (L/100km)")

# ----- Prediction -----
st.markdown("---")
if st.button("📈 Predict CO₂ Emission"):
    if missing_fields:
        st.error(f"⚠️ Please complete the following fields: {', '.join(missing_fields)}")
    else:
        vehicle_brand = brand_input if brand_input in top_brands else 'other'
        vehicle_model = model_input if model_input in top_models else 'other'

        input_df = pd.DataFrame({
            'Engine_Size_L': [engine_size],
            'Cylinders': [cylinders],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'vehicle_type': [vehicle_type],
            'Fuel_Consumption_Comb_mpg': [fuel_comb_mpg],
            'Fuel_Consumption_Comb_L_per_100_km': [fuel_comb_l],
            'vehicle_brand': [vehicle_brand],
            'vehicle_model': [vehicle_model]
        })

        prediction = model.predict(input_df)
        st.metric(label="Estimated CO₂ Emission", value=f"{prediction[0]:.2f} g/km")
