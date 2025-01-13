import pandas as pd
import joblib
import sqlalchemy
from dotenv import load_dotenv 
from sqlalchemy import create_engine 
import os

load_dotenv()
DB = os.getenv("DB")
engine = create_engine(DB)


def agg_data(data,groupby_col, agg_dict):
    pass

def lag_n(df: pd.DataFrame, group_col:str, lag_cols:list,sort_by: list, lag_number: int=1):
    """
    Function to calculate n lags from a given list of features
    
    args:
    df: pd.DataFrame - data frame to calculate lags on 
    group_col:str - name of column in data frame to group by lags 
    lag_cols:list - list of features to calculate lags 
    sort_by: list - list of features to sort data frame by
    lag_number: int - number of lags to calculate

    return data frame with lags added
    """ 
    pass

def load_model(path: str="models/classification_model.joblib"):
    """ Function to load model from path
    args: 
        path: str - location of the file with model.
    exception:
        Raise FileNotFoundError if there is no such file.
    return:
        scikit-learn model.
    """
    pass

def data_load(conditions: str,engine: sqlalchemy.engine.base.Engine = engine)-> pd.DataFrame:
    """
    Load data frame from postgres database.

    args:
        engine: sqlalchemy.engine.base.Engine - sql engine to connect with
        conditions: str - string with where clause.

    exceptions:
        Raise an exception when the data loading is not successful or the data frame has zero records.

    return:
        Loaded dataframe.
    """
    pass

def agg_df_to_hour_station(df: pd.DataFrame)-> pd.DataFrame:
    """
    Function to aggregate df loaded from SQL database to hourly data per station (departures and returns)

    args:
        df: pd.DataFrame - dataframe with columns to aggregate for classification model.
    return:
        aggregated and merged data frame.

    """
    pass

def agg_df_to_daily(df: pd.DataFrame)-> pd.DataFrame:
    """
    Function to aggregate df loaded from SQL database to daily data.

    args:
        df: pd.DataFrame - dataframe with columns to aggregate for regression model.
    return:
        aggregated data frame.

    """
    pass

def prepare_data_daily(df: pd.DataFrame)-> pd.DataFrame:
    """ Function to prepare data to make prediction for the next day.
    args: 
        df: pd.DataFrame - dataframe to prepare
    return:
        Data frame with added all needed features.
    """
    pass

def prepare_data(df: pd.DataFrame,lag_cols: list)-> pd.DataFrame:
    """
    Function to calculate metrics from initial df, so the prediction can be made

    returns df with added lags, daily aggregated and daily by station aggregated features
    """
    pass