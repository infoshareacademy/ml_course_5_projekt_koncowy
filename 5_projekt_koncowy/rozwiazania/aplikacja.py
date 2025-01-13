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
    while True:
        forecast_type= input(""" Jeżeli chcesz dokonać predyckji na kolejną godzinę wybierz 1. \n
                                Jeżeli chcesz dokonać predykcji na kolejny dzień wybierz 2.
                                W celu wylaczenia programu wybierz 0.""")
        if forecast_type =='0':
            break
        elif forecast_type=='1':
            print("Wybrano predykcję na kolejną godzinę.")
            forecast_hourly_station()
        elif forecast_type=='2':
            forecast_daily()
            print("Wybrano predykcję na kolejny dzień.")
        else:
            print('Podano złą liczbę.')

def forecast_daily():
    date = input("""Podaj datę dla której chcesz dokonać predykcji w formacie yyyy-mm-dd. \n
                     Predykcja będzie dokonana dla godzin 12:00 dnia poprzedniego do 11:59 dnia podanego.""")
    try:
        date = pd.to_datetime(date)
        yesterday = date - pd.Timedelta(days=1)
        cond = f"departure between '{yesterday}' and '{date}'"
    except TypeError:
        raise TypeError('Podano date w złym formacie.')
    df = data_load(cond)
    df = agg_df_to_daily(df)
    if (date - df.departure_date.max()).days==1:
        df.loc[df.index.max()+1,'departure_date'] = date
    df  = prepare_data_daily(df)
    model = load_model(path = MODEL_PATH_DAILY)
    df_to_pred= df.loc[df['ds']==date]
    preds = model.predict(df_to_pred[model.feature_names_in_])[0]
    print(f'Prognoza liczby wypożyczeń w okresie {yesterday} 12:00 a {date} 11:59 wynosi: {int(round(preds,0))}.')

def forecast_hourly_station():
    datetime = input("Podaj datę dla której chcesz dokonać predykcji w formacie yyyy-mm-dd HH:mm")
    try:
        datetime = pd.to_datetime(datetime)
        yesterday = datetime - pd.Timedelta(hours=25)
        cond = f"departure between '{yesterday}' and '{datetime}'"
    except TypeError:
        raise TypeError('Podano date w złym formacie.')
    df = data_load(cond)
    
    model = load_model(path = MODEL_PATH_HOURLY)
    encoder = load_model(path = ENCODER_PATH)
    agg_data = agg_df_to_hour_station(df)

    for i in agg_data.departure_name.unique():
        if len(agg_data.loc[(agg_data['departure_name']==i) &(agg_data['departure_date_hours']==datetime)])==0:
            agg_data.loc[(agg_data['departure_name']==i) &(agg_data['departure_date_hours']==datetime)] = [i,datetime]
        agg_data.loc[agg_data.index.max()+1,['departure_name','departure_date_hours']] = [i, datetime]
    df_to_pred = prepare_data(agg_data,lag_cols)
    df_to_pred =  df_to_pred[df_to_pred['departure_date_hours']==datetime]
    df_to_pred['departure_name_encoded'] = encoder.transform(df_to_pred[encoder.feature_names_in_])
    df_to_pred['pred'] = model.predict(df_to_pred[model.feature_names_in_])
    df_to_pred['pred_proba'] = model.predict_proba(df_to_pred[model.feature_names_in_])[:,1]
    pred_folder = PREDICTION_PATH
    if not os.path.exists(pred_folder):
        os.mkdir(pred_folder)
    #os.chdir(pred_folder)
    timestamp_name = str(datetime).replace(':','-')
    df_to_pred.to_csv(f'{pred_folder}/prediction_for_{timestamp_name}.csv')
    print(f"Predykcje zapisane w pliku {pred_folder}/prediction_for_{timestamp_name}.csv")
    while True:
        info= input("Czy chcesz wyświetlić predykcje dla danej stacji? (Y/N)?")
        if info.upper() !='Y':
            break
        station_name = input('Podaj nazwe stacji:')
        preds = df_to_pred.loc[df_to_pred['departure_name']==station_name,'pred_proba'].values
        if len(preds)==0:
            print('Podana stacja nie istnieje lub brak predykcji.')
        else:
            print(f'Prawdopododbieństwo braku rowerów na stacji w ciągu najbliższej godziny wynosi: {round(preds[0],3)}')