import requests
import pandas as pd
import numpy as np
from neuralprophet import load
from datetime import timedelta


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


def pjm_load(hrl, zone='AE'):
    #hrl = pd.read_csv(hrl_file)
    # hrl['ds'] = pd.to_datetime(hrl['datetime_beginning_ept'])
    # hrl = hrl.groupby(['ds', 'zone']).agg({'mw': 'sum'}).reset_index()
    start_date = hrl['ds'].min().strftime('%Y-%m-%d')
    end_date = (hrl['ds'].max() + timedelta(1)).strftime('%Y-%m-%d') # one day ahead

    zmap = pd.read_csv('Data/zone_mapping.csv')

    ## add weather
    idx = zmap['zone'].values.tolist().index(zone)
    wf1 = weather_forcast(zmap['lat'][idx], zmap['long'][idx], start_date, end_date)
    wf1['ds'] = pd.to_datetime(wf1['time'])
    wf1['zone'] = zmap['zone'][idx]
    hrl1 = hrl[hrl['zone'] == zone]
    dat1 = pd.merge(hrl1, wf1, on='ds', how='outer')
    dat1 = dat1.rename(
        columns={"mw": "y", "temperature_2m": "temp", "relativehumidity_2m": "rh", "surface_pressure": "pressure",
                "windspeed_10m": "windspeed", "snowfall": "snow"})
    dat1 = dat1[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
    dat1['rain'] = np.where(dat1['rain'] > 0, 1, 0)
    dat1['snow'] = np.where(dat1['snow'] > 0, 1, 0)

    return dat1

def load_model(zone, model='neuralprophet'):
    if model=='neuralprophet':
        model = load('model_checkpoints/np_f24_'+zone+'.pth')

    return model

def model_predict(zone, dat, model='neuralprophet'):
    m = load_model(zone, model)
    #dat = pjm_load(hrl, zone)
    ## prediction
    p1=m.predict(dat, raw=True, decompose=False)
    dat.y[-24:] = p1.iloc[-1].iloc[1:].astype('double')
    dat["source"] = "JPM"
    dat.source[-24:] = "pred"
    return dat

#pred = model_predict('AE')