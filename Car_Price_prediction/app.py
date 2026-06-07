import streamlit as st
import pandas as pd
import joblib
# Load the trained model
linmod = joblib.load('ford_price_prediction.pkl')
scaler = joblib.load('scaler.pkl')

# Derive categorical options from the scaler's expected feature names
model_options = [name[len('model_'):] for name in scaler.feature_names_in_ if name.startswith('model_')]
transmission_options = [name[len('transmission_'):] for name in scaler.feature_names_in_ if name.startswith('transmission_')]
fuel_options = [name[len('fuelType_'):] for name in scaler.feature_names_in_ if name.startswith('fuelType_')]

# Set the title of the app
st.title('Ford Car Price Prediction')   
# Create input fields for the features
year = st.slider('Year', min_value=1900, max_value=2024, value=2020)
mileage = st.number_input('Mileage', min_value=0, value=10000)
engine_size = st.number_input('Engine Size (L)', min_value=0.0, value=2.0, step=0.1)
model = st.selectbox('Model', model_options)
transmission = st.selectbox('Transmission', transmission_options)
fuel_type = st.selectbox('Fuel Type', fuel_options)
mpg = st.number_input('Miles Per Gallon (MPG)', min_value=0.0, value=30.0, step=0.1)
tax = st.number_input('Tax ($)', min_value=0, value=150)
# Create a button to make the prediction    
if st.button('Predict Price'):
    # Build the input row using the scaler's expected feature names
    expected_features = list(scaler.feature_names_in_)
    input_row = dict.fromkeys(expected_features, 0)
    input_row['year'] = year
    input_row['mileage'] = mileage
    input_row['engineSize'] = engine_size
    input_row['mpg'] = mpg
    input_row['tax'] = tax

    model_col = f'model_{model}'
    transmission_col = f'transmission_{transmission}'
    fuel_col = f'fuelType_{fuel_type}'

    if model_col not in input_row or transmission_col not in input_row or fuel_col not in input_row:
        st.error('Selected category is not recognized by the trained model.')
    else:
        input_row[model_col] = 1
        input_row[transmission_col] = 1
        input_row[fuel_col] = 1

        input_data = pd.DataFrame([input_row], columns=expected_features)
        scaled_input = scaler.transform(input_data)
        predicted_price = linmod.predict(scaled_input)
        st.subheader(f'Predicted Price: £{predicted_price[0]:.2f}')
    
        