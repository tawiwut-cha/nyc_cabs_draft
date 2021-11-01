'''
Helper data utility functions to help download and preprocess data
'''
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def drop_zero_records(df, cols:list=None):
    # drop records with 0 in the columns specified 
    if not cols:
        cols = df.select_dtypes('number').columns
    temp_df = df.copy()
    dtypes = dict(df.dtypes)
    for col in cols:
        temp_df.loc[temp_df[col]==0, [col]] = np.nan
        temp_df.dropna(axis=0, subset=[col], inplace=True)
    return temp_df.astype(dtypes) # int cols will get converted to float when replacing nulls

def drop_statistical_outliers(df, cols:list=None):
    # drop records with outlying values in the columns specified
    # outliers are further than 3 SDs from the mean
    if not cols:
        cols = df.select_dtypes('number').columns
    temp_df = df.copy()
    col_stats = {col:(temp_df[col].mean(), temp_df[col].std()) for col in cols}
    for col in cols:
        col_lower = col_stats[col][0] - 3*col_stats[col][1] 
        col_upper = col_stats[col][0] + 3*col_stats[col][1] 
        temp_df = temp_df[(temp_df[col] < col_upper) & (temp_df[col] > col_lower)]
    return temp_df

def drop_minmax(df, col, min_to_keep, max_to_keep):
    # drop values above max_to_keep or below min_to_keep
    return df[(df[col] <= max_to_keep) & (df[col] >= min_to_keep)]  
    
def haversine_distance(df, lat1, lat2, lon1, lon2):
    '''
    Simple distance calculator, assume angle is so small since we are only in new york city
    Returns distance in km

    lat1, lat2, lon1, lon2 are col names in df
    '''
    # make a copy
    temp_df = df.copy()
    # The math module contains a function named radians which converts from degrees to radians.
    lat1_rad = np.vectorize(np.math.radians)(df[lat1])
    lat2_rad = np.vectorize(np.math.radians)(df[lat2])
    lon1_rad = np.vectorize(np.math.radians)(df[lon1])
    lon2_rad = np.vectorize(np.math.radians)(df[lon2])

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.vectorize(np.math.sin)(dlat / 2)**2 + \
        np.vectorize(np.math.cos)(lat1_rad) * np.vectorize(np.math.cos)(lat2_rad) * np.vectorize(np.math.sin)(dlon / 2)**2
 
    c = 2 * np.vectorize(np.math.asin)(np.sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    temp_df['trip_distance'] = c*r 
    return temp_df  

def feature_eng_df(df):
    '''
    Feature engineering 
    - Create distance column
    - Decompose datetime features
    - Log transform target
    '''
    # create distance column
    df = haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude')
    # decompose date time features
    df['pickup_datetime_month'] = df['pickup_datetime'].dt.month
    df['pickup_datetime_date'] = df['pickup_datetime'].dt.day
    df['pickup_datetime_day_of_week'] = df['pickup_datetime'].dt.day_of_week
    df['pickup_datetime_hour'] = df['pickup_datetime'].dt.hour
    # log transform target
    df['log_trip_duration'] = np.log(df['trip_duration'])

    return df

def preprocess_df(df):
    '''
    df --> X, y
    '''
    # Select features for X
    X_cols_num = [
        'trip_distance',
        # 'pickup_datetime_month',
        'pickup_datetime_date',
        'pickup_datetime_day_of_week',
        'pickup_datetime_hour',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude',
        ]
    X_cols_cat = []
    X_cols = X_cols_num + X_cols_cat

    num_pipeline = Pipeline(
        steps=[
            ('median_imputer', SimpleImputer(strategy='median')),
            ('standard_scaler', StandardScaler()),
        ])

    preprocessing_pl = Pipeline(
        steps=[
            ('selector', FunctionTransformer(lambda df: df[X_cols])),
            ('column_transformer', ColumnTransformer([('num', num_pipeline, X_cols_num),])),
        ])        

    X = preprocessing_pl.fit_transform(df)

    df_for_model = pd.DataFrame(X, columns=X_cols)
    df_for_model['log_trip_duration'] = df['log_trip_duration'].values
    return df_for_model

if __name__ == '__main__':
    SOURCE_FILEPATH = os.path.join(os.pardir, 'datasets', 'train.csv')
    DESTINATION_FILEPATH = os.path.join(os.pardir, 'datasets', 'train_preprocessed_streamlit.csv')

    if not os.path.isfile(DESTINATION_FILEPATH):
        # Load data
        print('Loading data...')
        df = pd.read_csv(SOURCE_FILEPATH, parse_dates=['pickup_datetime', 'dropoff_datetime'])

        # Clean data
        print('Cleaning data...')
        ## remove zero passenger
        df = drop_zero_records(df, ['passenger_count'])
        ## remove statistical outliers
        df = drop_statistical_outliers(df)
        ## remove further outliers by min max
        NYC_MIN_LON, NYC_MAX_LON = -74.4, -73.4 # approx from google map
        NYC_MIN_LAT, NYC_MAX_LAT = 40, 41.6 # approx from google map
        df = drop_minmax(df, 'pickup_latitude', NYC_MIN_LAT, NYC_MAX_LAT)
        df = drop_minmax(df, 'pickup_longitude', NYC_MIN_LON, NYC_MAX_LON)
        df = drop_minmax(df, 'dropoff_latitude', NYC_MIN_LAT, NYC_MAX_LAT)
        df = drop_minmax(df, 'dropoff_longitude', NYC_MIN_LON, NYC_MAX_LON)

        # Feature engineering
        print('Creating features...')
        df = feature_eng_df(df)

        # Preprocess data
        print('Preprocessing data...')
        df = preprocess_df(df)

        # Write to csv
        df.to_csv(DESTINATION_FILEPATH, index=False)

    print('Data \'datasets/train_streamlit.csv\' ready to feed into ML model.')