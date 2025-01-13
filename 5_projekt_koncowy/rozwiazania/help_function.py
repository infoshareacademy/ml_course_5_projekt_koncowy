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
    return data.groupby(groupby_col).agg(agg_dict).reset_index()

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
    df = df.sort_values(by=sort_by)
    for i in lag_cols:
        df[f'{i}_lag_{lag_number}'] = df.groupby(group_col)[i].shift(lag_number)
    return df

def load_model(path: str="models/classification_model.joblib"):
    """ Function to load model from path
    args: 
        path: str - location of the file with model.
    exception:
        Raise FileNotFoundError if there is no such file.
    return:
        scikit-learn model.
    """
    try: 
        model = joblib.load(path)
        print('Model zaladowany')
        return model
    except FileNotFoundError:
        print('Model nie znaleziony')

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
    try: 
        df = pd.read_sql(f"""select * from renting_data where {conditions} """, con=engine)
        if df.shape[0]==0:
            raise BaseException('Brak danych dla zadanych warunków.')
    except:
        raise BaseException('Nie udało się pobrać danych.')
    return df

def agg_df_to_hour_station(df: pd.DataFrame)-> pd.DataFrame:
    """
    Function to aggregate df loaded from SQL database to hourly data per station (departures and returns)

    args:
        df: pd.DataFrame - dataframe with columns to aggregate for classification model.
    return:
        aggregated and merged data frame.

    """
    df['departure'] = pd.to_datetime(df['departure'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    df['departure_date_hours'] = pd.to_datetime(df['departure_date_hours'])
    df['return'] = pd.to_datetime(df['return'])
    df['return_date'] = pd.to_datetime(df['return_date'])
    df['return_date_hours'] = pd.to_datetime(df['return_date_hours'])
    ## dodac weryfikacje kolumn
    cols = ['departure_name',
            'departure_id',
            'distance (m)',
            'duration (sec.)',
            'avg_speed (km/h)',
            'Air temperature (degC)',
            'return_name',
            'return_id'  ]
    if not set(cols).issubset(df.columns):
        raise KeyError('Ramka danych nie zawiera wymaganych kolumn.')
    df_agg_dep = agg_data(df, ['departure_name', 'departure_date_hours'],
                      {'departure_id': 'count',
                         'distance (m)': 'mean',
                         'duration (sec.)': 'mean',
                         'avg_speed (km/h)': 'mean',
                         'Air temperature (degC)': 'mean'})
    df_agg_dep = df_agg_dep.rename(columns = {'departure_id': 'numbers_of_departures'})

    df_agg_ret = agg_data(df, ['return_name', 'return_date_hours'],
                      {'return_id': 'count',
                         'distance (m)': 'mean',
                         'duration (sec.)': 'mean',
                         'avg_speed (km/h)': 'mean',
                         'Air temperature (degC)': 'mean'})
    
    df_agg_ret  = df_agg_ret.rename(columns = {'return_id': 'number_of_returns'})

    df_merged = df_agg_dep.merge(df_agg_ret,
                         left_on = ['departure_name','departure_date_hours'],
                         right_on=['return_name','return_date_hours'],
                         how = 'outer',
                         suffixes=('_dep','_ret'))
    df_merged['departure_name'] = df_merged['departure_name'].fillna(df_merged['return_name'])
    df_merged['departure_date_hours'] = df_merged['departure_date_hours'].fillna(df_merged['return_date_hours'])
    df_merged['Air temperature (degC)'] = df_merged['Air temperature (degC)_dep'].fillna(df_merged['Air temperature (degC)_ret'])
    del df_merged['return_date_hours']
    del df_merged['return_name']
    del df_merged['Air temperature (degC)_dep']
    del df_merged['Air temperature (degC)_ret']
    
    return df_merged

def agg_df_to_daily(df: pd.DataFrame)-> pd.DataFrame:
    """
    Function to aggregate df loaded from SQL database to daily data.

    args:
        df: pd.DataFrame - dataframe with columns to aggregate for regression model.
    return:
        aggregated data frame.

    """
    df['departure'] = pd.to_datetime(df['departure'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])

    cols = ['departure_date',
            'departure_name',
            'distance (m)',
            'duration (sec.)',
            'avg_speed (km/h)',
            'Air temperature (degC)'
            ]                
    if not set(cols).issubset(df.columns):
        raise KeyError('Ramka danych nie zawiera wymaganych kolumn.')                      
    df_total_agg  =agg_data(df,['departure_date'],
                        {'departure_name': 'count',
                         'distance (m)': 'mean',
                         'duration (sec.)': 'mean',
                         'avg_speed (km/h)': 'mean',
                         'Air temperature (degC)': 'mean'})
    df_total_agg = df_total_agg.rename(columns = {'departure_name': 'numbers_of_renting'})
    return df_total_agg

def prepare_data_daily(df: pd.DataFrame)-> pd.DataFrame:
    """ Function to prepare data to make prediction for the next day.
    args: 
        df: pd.DataFrame - dataframe to prepare
    return:
        Data frame with added all needed features.
    """
    df = df.set_index('departure_date')
    df_resampled = df.resample('D').sum().fillna(0)
    df_resampled = df_resampled.sort_index()
    df_resampled['temperature_yesterday'] = df_resampled['Air temperature (degC)'].shift(1)
    df_resampled['avg_speed_yesterday'] = df_resampled['avg_speed (km/h)'].shift(1)
    df_resampled['avg_duration_yesterday'] = df_resampled['duration (sec.)'].shift(1)
    df_resampled = df_resampled.rename(columns = {'numbers_of_renting':'y' })
    df_resampled['y_yesterday'] = df_resampled['y'].shift(1)
    # dodanie informacji z daty
    df_resampled['month'] = df_resampled.index.month
    df_resampled['day'] = df_resampled.index.day
    df_resampled['quarter'] = df_resampled.index.quarter
    # usunięcie braków danych
    df_resampled = df_resampled.dropna().reset_index()
    df_resampled = df_resampled.rename(columns = {'departure_date':'ds'})
    return df_resampled

def prepare_data(df: pd.DataFrame,lag_cols: list)-> pd.DataFrame:
    """
    Function to calculate metrics from initial df, so the prediction can be made

    returns df with added lags, daily aggregated and daily by station aggregated features
    """
    df = df.set_index('departure_date_hours').groupby('departure_name').resample('h').mean().reset_index()
    df['Air temperature (degC)'] = df['Air temperature (degC)'].fillna(-999)
    df =df.fillna(0)
    df['y_cat'] = ((df['numbers_of_departures']-1)> df['number_of_returns']).astype(int)
    if not set(lag_cols).issubset(df.columns):
        raise KeyError('Given dataframe does not contain required fields.')
    df['hour'] = df['departure_date_hours'].dt.hour
    df['day'] = df['departure_date_hours'].dt.day
    df['month'] = df['departure_date_hours'].dt.month
    df['quarter'] = df['departure_date_hours'].dt.quarter
    for i in [1,2,3,6,9,12,24]:
        df = lag_n(df = df,
                      group_col='departure_name',
                      lag_cols=lag_cols,
                      sort_by='departure_date_hours',
                      lag_number=i)
    df.loc[df['Air temperature (degC)']==-999,'Air temperature (degC)'] = df.loc[df['Air temperature (degC)']==-999,
                                                                                                  'Air temperature (degC)_lag_1']
    df.loc[df['Air temperature (degC)']==-999,'Air temperature (degC)'] = df.loc[df['Air temperature (degC)']==-999,
                                                                                                  'Air temperature (degC)_lag_2']
    return df