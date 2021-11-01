'''Create an NYC taxi trip duration web-application based on your Kaggle competition model

By: Tawiwut Charuwat
Start date: 29/10/2021
'''

# imports
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from preprocess_data_for_streamlit import *

st.write('''
# NYC Taxi trip duration prediction!

Predict your taxi trip duration by entering:
- pickup location
- dropoff location

''')

# Load preprocessed data
@st.cache
def load_data():
    return pd.read_csv('../datasets/train_preprocessed_streamlit.csv')
df = load_data()
st.dataframe(df.head())

# Get user input
def user_input_features():
    pickup_date = st.date_input('Pickup date')
    pickup_time = st.time_input('Pickup time')
    pickup_lat = st.number_input('pickup latitude', min_value=40.0000, max_value=42.0000, value=40.7580)
    pickup_lon = st.number_input('pickup longitude', min_value=-74.5000, max_value=-73.5000, value=-73.9855)
    dropoff_lat = st.number_input('dropoff latitude', min_value=40.0000, max_value=42.0000, value=40.7800)
    dropoff_lon = st.number_input('dropoff longitude', min_value=-74.5000, max_value=-73.5000, value=-73.6900)

    input_df = pd.DataFrame(
        columns = [
            'trip_distance',
            'pickup_datetime_date',
            'pickup_datetime_day_of_week',
            'pickup_datetime_hour',
            'pickup_latitude',
            'pickup_longitude',
            'dropoff_latitude',
            'dropoff_longitude',
    ]) 

    input_df['pickup_datetime_date'] = [pickup_date.day]
    input_df['pickup_datetime_day_of_week'] = [pickup_date.weekday()]
    input_df['pickup_datetime_hour'] = [pickup_time.hour]
    input_df['pickup_latitude'] = [pickup_lat]
    input_df['pickup_longitude'] = [pickup_lon]
    input_df['dropoff_latitude'] = [dropoff_lat]
    input_df['dropoff_longitude'] = [dropoff_lon]
    
    # for feeding to model
    trip_df = pd.DataFrame({'lat':[pickup_lat, dropoff_lat], 'lon':[pickup_lon,dropoff_lon]}) # for showing trip on map

    return input_df, trip_df #, pickup_date, pickup_time

input_df, trip_df = user_input_features()

# Show trip on a map
st.map(trip_df)

# Preprocess input data
## Feature engineering
input_df = haversine_distance(input_df, 'pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude')
# st.dataframe(input_df)
## Pass through preprocessing pipeline
preprocess_pl = load('../competition/preprocessing_pl.joblib')
X = preprocess_pl.transform(input_df)
# st.table(X)

# Fancy button
if st.button('Estimate trip duration'):
    # Load model
    model = load('../competition/best_estimator.joblib')
    # Predict using input data --> log scale
    predictions = model.predict(X) 
    # Convert predictions to minutes (rounded up)
    predictions = np.exp(predictions) // 60 + 1
    st.write(f'Your trip will take about {int(predictions[0])} minutes')
    




