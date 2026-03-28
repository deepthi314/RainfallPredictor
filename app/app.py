import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Manila Rainfall Predictor", layout="wide")

# Cache Models
@st.cache_resource
def load_models():
    model = joblib.load(r"e:\college\WaterHarvesting\rainfall_prediction\models\best_model.pkl")
    scaler = joblib.load(r"e:\college\WaterHarvesting\rainfall_prediction\models\scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you've run the training script.")
    st.stop()

st.title("🌧️ Manila Rainfall Predictor")
st.markdown("""
Predict daily rainfall for Manila, Philippines based on NASA POWER meteorological parameters. 
Location: 14.5822°N, 120.9751°E.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Meteorological Parameters")
    
    allsky_sw_dwn = st.slider("Solar Shortwave Irradiation (kW-hr/m2/day)", 0.0, 10.0, 5.0)
    allsky_sw_dni = st.slider("Direct Normal Irradiation (kW-hr/m2/day)", 0.0, 10.0, 4.0)
    t2m = st.number_input("Temperature at 2m (°C)", -10.0, 50.0, 27.0)
    t2m_dew = st.number_input("Dew Point at 2m (°C)", -10.0, 50.0, 23.0)
    t2m_wet = st.number_input("Wet Bulb Temp at 2m (°C)", -10.0, 50.0, 24.0)
    t2m_max = st.number_input("Max Temperature at 2m (°C)", -10.0, 50.0, 31.0)
    t2m_min = st.number_input("Min Temperature at 2m (°C)", -10.0, 50.0, 24.0)
    rh2m = st.slider("Relative Humidity at 2m (%)", 0.0, 100.0, 80.0)
    qv2m = st.number_input("Specific Humidity (g/kg)", 0.0, 50.0, 18.0)
    ws2m = st.slider("Wind Speed at 2m (m/s)", 0.0, 30.0, 2.5)
    gwettop = st.slider("Surface Soil Moisture", 0.0, 1.0, 0.6)

if st.button("Predict Rainfall"):
    # Prepare input
    # Note: The model expects specific features include MONTH, WEEK, SEASON, and LAGS
    # For a simple web app, we might need to assume default or today's date for lags
    # However, the user request lists sliders for all 11 basic features.
    
    # Feature columns expected by the model
    feature_cols = [
        'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'T2M', 'T2MDEW', 'T2MWET', 
        'T2M_MAX', 'T2M_MIN', 'RH2M', 'QV2M', 'WS2M', 'GWETTOP', 
        'MONTH', 'WEEK_OF_YEAR', 'SEASON', 'LAG_1', 'LAG_7', 'LAG_30', 
        'ROLLING_7_MEAN', 'ROLLING_30_MEAN'
    ]
    
    # For a simple web app, we'll set lags and rolling to 0 and time features to default mid-year
    input_data = pd.DataFrame([{
        'ALLSKY_SFC_SW_DWN': allsky_sw_dwn,
        'ALLSKY_SFC_SW_DNI': allsky_sw_dni,
        'T2M': t2m,
        'T2MDEW': t2m_dew,
        'T2MWET': t2m_wet,
        'T2M_MAX': t2m_max,
        'T2M_MIN': t2m_min,
        'RH2M': rh2m,
        'QV2M': qv2m,
        'WS2M': ws2m,
        'GWETTOP': gwettop,
        'MONTH': 6,          # Added missing column
        'WEEK_OF_YEAR': 26, 
        'SEASON': 3, 
        'LAG_1': 0.0,
        'LAG_7': 0.0,
        'LAG_30': 0.0,
        'ROLLING_7_MEAN': 0.0,
        'ROLLING_30_MEAN': 0.0
    }])
    
    # Reorder columns to match feature_cols exactly
    input_data = input_data[feature_cols]
    
    # Scaling (using the scaler fitted on original 11 features)
    # The scaler was fitted on ['ALLSKY_SFC_SW_DWN', ..., 'GWETTOP']
    # We should scale only those
    feature_names_scaling = ['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'QV2M', 'WS2M', 'GWETTOP']
    input_data[feature_names_scaling] = scaler.transform(input_data[feature_names_scaling])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction) # Rainfall cannot be negative
    
    with col2:
        st.header("Prediction Results")
        
        # Color based on severity
        if prediction < 5:
            severity = "Low"
            color = "green"
        elif prediction < 20:
            severity = "Moderate"
            color = "orange"
        elif prediction < 50:
            severity = "Heavy"
            color = "red"
        else:
            severity = "Extreme"
            color = "maroon"
            
        st.markdown(f"### Predicted Rainfall: <span style='color:{color}'>{prediction:.2f} mm/day</span>", unsafe_allow_html=True)
        st.info(f"Interpretation: **{severity} Rainfall**")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Rainfall (mm/day)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps' : [
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 20], 'color': "yellow"},
                    {'range': [20, 50], 'color': "orange"},
                    {'range': [50, 100], 'color': "red"}],
            }
        ))
        st.plotly_chart(fig)

st.sidebar.info("Model: The best performing model from Random Forest, XGBoost, and others trained on years 1995-2020.")
