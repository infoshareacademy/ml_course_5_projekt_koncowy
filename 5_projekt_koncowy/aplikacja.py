from help_function import load_model, data_load,  agg_df_to_daily, prepare_data_daily,agg_df_to_hour_station, prepare_data
import pandas as pd
import os
from dotenv import load_dotenv 

load_dotenv()
MODEL_PATH_DAILY = os.getenv("MODEL_PATH_DAILY")
MODEL_PATH_HOURLY = os.getenv("MODEL_PATH_HOURLY")
ENCODER_PATH = os.getenv("ENCODER_PATH")
PREDICTION_PATH = os.getenv("PREDICTION_PATH")
lag_cols = ['numbers_of_departures',
       'distance (m)_dep', 'duration (sec.)_dep', 'avg_speed (km/h)_dep',
       'number_of_returns', 'distance (m)_ret', 'duration (sec.)_ret',
       'avg_speed (km/h)_ret', 'Air temperature (degC)', 'y_cat']

def forecast_process():
    """
    Function to calculate predictions:
        1. Create probability of decreasing for all stations for given date and hour.
        2. Save results in the given location.
        3. Create predictions of numbers of departures in the next day.
    """
    pass

def forecast_daily():
    """
    Function to create daily forecasts of rentings.
    """
    pass

def forecast_hourly_station():
    """
    Function to calculate probability of decreasing for all stations for given date and hour
    """
    pass