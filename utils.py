import requests
import pandas as pd
from neuralprophet import load

def weather_forcast(lat, lon, start_date, end_date):
    url_api = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat, #41.85,
        'longitude': lon, #-87.65,
        'hourly[]': ['temperature_2m','relativehumidity_2m','precipitation','surface_pressure','windspeed_10m','rain','snowfall'],
        'timezone': 'auto',
        'temperature_unit': 'fahrenheit',
        'start_date': start_date,
        'end_date': end_date
    }
    try:
        req = requests.get(url_api, params)
    except:
        print("Err from weather forcasting")

    wf = pd.DataFrame(req.json()['hourly'])
    wf.insert(loc=0, column='long', value=lon)
    wf.insert(loc=0, column='lat', value=lat)
    return wf


def pjm_load(zone, start_date, end_date):
    pass


def load_model(zone, model='neuralprophet'):
    if model=='neuralprophet':
        model = load('model_checkpoints/np_f24_'+zone+'.pth')

    return model

def model_predict(model):
    pass