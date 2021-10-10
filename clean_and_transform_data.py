
import os
from urllib.request import urlretrieve
import pandas as pd
import numpy as np

DOWNLOAD_ROOT = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'
TRIPDATA_URL = DOWNLOAD_ROOT+'(vehicle_type)_tripdata_(YYYY)-(MM).csv'
TRIPDATA_PATH = os.path.join('datasets', '(vehicle_type)_tripdata_(YYYY)-(MM).csv')

VEHICLE_TYPES = ['yellow', 'green', 'fhv', 'fhvhv']
YEAR_MONTH_PAIRS = [('2019-06'),('2019-05')]

# fetch data
def fetch_tripdata(year_month_pairs:list=YEAR_MONTH_PAIRS, vehicle_types:list=VEHICLE_TYPES, tripdata_url=TRIPDATA_URL, tripdata_path=TRIPDATA_PATH):
    '''
    download tripdata from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

    Params:
    year_month_pairs - list of tuple of year month ex [('2020-05')]
    vehicle_types - list of vehicle type data to download ['yellow']
    tripdata_url - url for data download got from webpage
    tripdata_path - path to place downloaded files
    '''
    print('Fetching tripdata ......')
    for year_month_pair in year_month_pairs:
        year_, month_ = year_month_pair.split('-')[0], year_month_pair.split('-')[1] 
        for vehicle_type in vehicle_types:
            # replace formatting for vehicle_type, year, month
            url_ = tripdata_url.replace('(vehicle_type)', vehicle_type).replace('(YYYY)', year_).replace('(MM)', month_)
            path_ = tripdata_path.replace('(vehicle_type)', vehicle_type).replace('(YYYY)', year_).replace('(MM)', month_)           
            if not os.path.isfile(path_):
                print(f'{path_} not downloaded')
                print(f'Downloading from {url_}')
                urlretrieve(url_, path_)
            else:
                print(f'{path_} downloaded already')
    print('################ DONE #################')

def load_tripdata(year_month_pairs:list=YEAR_MONTH_PAIRS, vehicle_types:list=VEHICLE_TYPES, tripdata_path=TRIPDATA_PATH):
    dfs = dict()
    print('Loading tripdata ......')
    for year_month_pair in year_month_pairs:
        year_, month_ = year_month_pair.split('-')[0], year_month_pair.split('-')[1] 
        for vehicle_type in vehicle_types:
            path_ = tripdata_path.replace('(vehicle_type)', vehicle_type).replace('(YYYY)', year_).replace('(MM)', month_)
            dfs[f'{year_month_pair}-{vehicle_type}'] = pd.read_csv(path_)
    print('################ DONE #################')
    return dfs

def combine_all_tripdata(dfs:dict):
    '''
    Combine yellow, green, fhv, fhvhv tripdata by common features
    - pickup_datetime
    - dropoff_datetime
    - pickup_locationid
    - dropoff_locationid

    Also add tags
    - vehicle_type
    '''
    return_col_names = ['pickup_datetime', 'dropoff_datetime', 'pickup_locationid', 'dropoff_locationid', 'trip_distance', 'vehicle_type']
    return_df = pd.DataFrame(data={_:[] for _ in return_col_names})
    for dataset_name, df in dfs.items():
        if 'yellow' in dataset_name:
            append_df = df.loc[:, ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']]
            append_df['vehicle_type'] = 'yellow'
        elif 'green' in dataset_name:
            append_df = df.loc[:, ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']]
            append_df['vehicle_type'] = 'yellow'
        elif 'fhvhv' in dataset_name:
            append_df = df.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']]
            append_df['trip_distance'] = np.nan
            append_df['vehicle_type'] = 'fhvhv'
        elif 'fhv' in dataset_name:
            append_df = df.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']]
            append_df['trip_distance'] = np.nan
            append_df['vehicle_type'] = 'fhv'
        append_df.columns = return_col_names
        return_df = return_df.append(append_df, ignore_index=True)
    return return_df



fetch_tripdata()
dfs = load_tripdata()
combined_tripdata = combine_all_tripdata(dfs)
combined_tripdata.to_csv(os.path.join('datasets', 'combined_tripdata.csv'), index=False)


