import pandas as pd
import joblib
import sqlalchemy
def agg_data(data,groupby_col, agg_dict):
    return data.groupby(groupby_col).agg(agg_dict).reset_index()

def lag_n(df: pd.DataFrame, group_col:str, lag_cols:list,sort_by: list, lag_number=1):
    df = df.sort_values(by=sort_by)
    for i in lag_cols:
        df[f'{i}_lag_{lag_number}'] =df.groupby(group_col)[i].shift(lag_number)
    return df

def load_model(path: str="model_restaurant_revenue.joblib"):
    try:
        model = joblib.load(path)
        print('Model zaladowany')
        return model
    except FileNotFoundError:
        print('Model nie znaleziony')

def data_load(engine: sqlalchemy.engine.base.Engine, conditions: str):
    try:
     to_pred =  pd.read_sql(f"""select * from  bikes_renting_data where {conditions}""", con=engine )
    except:
       print("Nie udało się pobrać danych dla zadanych warunków")
    if to_pred.shape[0] ==0:
       raise BaseException("Brak danych dla podanych ograniczeń")
    return to_pred

def agg_loaded_df(df):
    df['departure'] = pd.to_datetime(df['departure'])
    df["departure_date"] = df['departure'].dt.round("D")
    df["departure_date_hours"] = df['departure'].dt.round("h")
    df['return'] = pd.to_datetime(df['return'])
    df["return_date"] = df['return'].dt.round("D")
    df["return_date_hours"] = df['return'].dt.round("h")

    cols = ['departure_id',
            'departure_name',
            'Air temperature (degC)',
            'distance (m)',
            'duration (sec.)',
            'return_id']
    if not cols in df.columns:
        raise KeyError('Given dataframe does not contain erquired fields.')
    df_agg_dep= agg_data(df,
                    ['departure_id','departure_date_hours'],
                    {'departure_name':'count',
                    'Air temperature (degC)': 'mean',
                    'distance (m)': 'mean',
                    'duration (sec.)':'mean'}  )
    df_agg_dep = df_agg_dep.rename(columns={'departure_name': 'nr_of_departures'})

    df_agg_ret= agg_data(df,
                    ['return_id','return_date_hours'],
                    {'return_name':'count',
                    'Air temperature (degC)': 'mean',
                    'distance (m)': 'mean',
                    'duration (sec.)':'mean'}  )
    
    df_agg_ret = df_agg_ret.rename(columns={'return_name':'nr_of_returns'})
    df_merged = df_agg_dep.merge(df_agg_ret[['return_date_hours','return_id','nr_of_returns']], left_on =[
    'departure_id', 'departure_date_hours'], right_on=['return_id','return_date_hours'], how='left')
    df_merged['nr_of_returns'] = df_merged['nr_of_returns'].fillna(0)
    return df_merged

def prepare_data(df):

    lag_cols = ['nr_of_departures','nr_of_returns','Air temperature (degC)','distance (m)','duration (sec.)']
    if not lag_cols in df.columns:
        raise KeyError('Given dataframe does not contain erquired fields.')
    agg_dict = {
    'nr_of_departures': 'sum',
    'Air temperature (degC)': 'mean',
    'distance (m)': 'mean',
    'duration (sec.)': 'mean'}
    df = lag_n(df, 
                  group_col='departure_id',
                  sort_by = ['departure_id','departure_date_hours'],
                  lag_cols=lag_cols)
    df["departure_date"] = df['departure_date_hours'].dt.date
    daily_data = agg_data(df,'departure_date',agg_dict)
    daily_data.columns = ['yt_' + i for i in daily_data.columns]
    df["yesterday_date"] = df['departure_date'] - pd.Timedelta(days=1)

    df = df.merge(
    daily_data, left_on = 'yesterday_date', right_on='yt_departure_date', how='left').fillna(0)

    daily_data_station = agg_data(df,['departure_date','departure_id'],agg_dict)
    daily_data_station.columns = ['yt_station_' + i for i in daily_data_station.columns]

    df = df.merge(
    daily_data_station, 
    left_on = ['yesterday_date','departure_id'], 
    right_on=['yt_station_departure_date','yt_station_departure_id'], 
    how='left').fillna(0)

    return df




