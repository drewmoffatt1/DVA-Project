import pandas as pd
import numpy as np
import requests
import json
import datetime as dt
import xgboost as xgb
import sklearn

def get_date():
    return (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')

def zone_mapping():
    zm = pd.read_csv('models/zone_mapping_hist_peak.csv')
    zone_dict = dict()
    for i in zm['zone']:
        this_val = dict()
        this_val['lat'] = zm['lat'].loc[zm['zone'] == i].iloc[0]
        this_val['long'] = zm['long'].loc[zm['zone'] == i].iloc[0]
        zone_dict[i] = this_val
    return zone_dict

def holiday_cal(strdate):
    from pandas.tseries.holiday import get_calendar, nearest_workday, Holiday

    usfh = get_calendar("USFederalHolidayCalendar")
    juneteenth = Holiday(
        "Juneteenth", month=6, day=19, start_date='2021-10-01', observance=nearest_workday
    )
    if not any(h.name == "Juneteenth" for h in usfh.rules):
        usfh.rules.append(juneteenth)
    return usfh.holidays(strdate, strdate)

def cycle_encode(df, var, minval, maxval):
    space = 2*np.pi/(maxval-minval)
    return np.cos(space*df[var]), np.sin(space*df[var])

def get_dow_encode(row):
    dow_dict = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    return dow_dict.get(row['dow'])

def get_season_encode(row):
    if row['month'] in [6, 7, 8, 9]:
        return 1
    elif row['month'] in [5, 10]:
        return 2
    elif row['month'] in [4, 11]:
        return 3
    else:
        return 4
    
def make_empty_df():
    ## bring in date
    strdate = get_date()
    
    ## bring in holidays
    holidays = holiday_cal(strdate)
    
    ## create initial DF
    df = pd.DataFrame()
    df['datetime'] = pd.date_range(strdate, periods=24, freq="H")
    df['datetime'] = df['datetime'].dt.tz_localize('US/Eastern', ambiguous = 'NaT', nonexistent = 'NaT')
    df = df.dropna()
    
    ## basic date variables
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    ## extract DOW
    df['dow'] = df['datetime'].dt.day_name()

    ## extract DST
    df['is_dst'] = (df['datetime'].dt.strftime('%z') == '-0400')

    ## extract stay-at-home orders timeframe
    sah_start = pd.to_datetime('3/18/2020').tz_localize('US/Eastern', ambiguous = 'NaT')
    sah_end = pd.to_datetime('6/30/2020').tz_localize('US/Eastern', ambiguous = 'NaT')
    df['is_sah'] = df['datetime'].between(sah_start, sah_end)

    ## extract pre-covid
    df['precovid'] = df['datetime'] < sah_start

    ## extract post-covid
    df['postcovid'] = df['datetime'] > sah_end
    
    ## weekend indicator
    df['weekend'] = df['dow'].isin(['Sunday', 'Saturday'])
    
    ## recent
    df['recent'] = 1
    
    ## create cyclical month and hour variables
    df['hour_cos'], df['hour_sin'] = cycle_encode(df, 'hour', 0, 23)
    df['month_cos'], df['month_sin'] = cycle_encode(df, 'month', 1, 12)    
    
    ## create dow encoding
    df['dow_num'] = df.apply(get_dow_encode, axis = 1)
    
    ##holiday
    df['holiday'] = df['datetime'].dt.strftime('%Y-%m-%d').isin(holidays)
    
    ## season encoding
    df['season_num'] = df.apply(get_season_encode, axis = 1)
    
    return df[['datetime', 'date', 'hour', 'year', 'dow', 'is_dst', 'is_sah',
       'precovid', 'postcovid', 'weekend', 'recent', 'hour_cos', 'hour_sin',
       'month_cos', 'month_sin', 'dow_num', 'holiday', 'season_num']].copy()

def api_weather_forecast(zone, zones_dict):
    ## Set parameters for API call
    latitude = str(zones_dict.get(zone).get('lat'))
    longitude = str(zones_dict.get(zone).get('long'))
    strdate = get_date()
    timezone = 'America%2FNew_York'

    ## Make API call
    response = requests.get(
                            "https://api.open-meteo.com/v1/forecast?" + \
                            'latitude=' + latitude + \
                            '&longitude=' + longitude + \
                            '&hourly=temperature_2m,relativehumidity_2m,precipitation,weathercode,surface_pressure,' + \
                            'windspeed_10m' + \
                            '&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch' + \
                            '&timezone=America%2FNew_York' + \
                            '&start_date=' + strdate + \
                            '&end_date=' + strdate,verify=False
                           )

    forecast = response.json()
    tzz = forecast.get('timezone_abbreviation')
    if 'S' in tzz:
        affix = '-0500'
    else:
        affix = '-0400'


    ## Format API response

    # Define weathercodes as rain/snow
    rain_codes = [51,53,55,61,63,65,80,81,82,95,96,99]
    snow_codes = [56,57,66,67,71,73,75,77,85,86]

    hourly = forecast.get('hourly')

    df2 = pd.DataFrame()
    for i in hourly.keys():
        df2[i] = hourly.get(i)
    df2['datetime'] = pd.to_datetime(df2['time']+affix)
    df2['datetime'] = df2['datetime'].dt.tz_convert('US/Eastern')
    df2 = df2.dropna()
    df2['rain'] = df2['weathercode'].isin(rain_codes)
    df2['snow'] = df2['weathercode'].isin(snow_codes)
    
    ## convert pressure from hPa to inHg
    df2['pressure'] = df2['surface_pressure'] * 0.029529983071445
    df2 = df2[['temperature_2m', 'relativehumidity_2m', 'precipitation',
           'pressure', 'windspeed_10m', 'datetime', 'rain',
           'snow']].copy()
    df2.columns = ['temp', 'rh', 'precip',
           'pressure', 'windspeed', 'datetime', 'rain',
           'snow']
    return df2

def create_analysis_df(zone, zones_dict, base_df):
    df2 = api_weather_forecast(zone, zones_dict)
    df8 = base_df.merge(df2, on='datetime', how='left')
    result_df = df8[['date', 'hour', 'temp']].copy()
    result_df['zone'] = zone
    historical_df = df8[['date', 'hour', 'temp', 'precip','pressure','windspeed','rain','snow']].copy()
    historical_df['zone'] = zone
    x = df8[['year', 'weekend', 'holiday', 'is_dst', 'is_sah', 'precovid',
       'postcovid', 'temp', 'precip', 'rh', 'pressure', 'windspeed', 'rain', 'snow',
       'recent', 'hour_cos', 'hour_sin', 'month_cos', 'month_sin', 'dow_num']].copy()
    return result_df, x, historical_df

def get_hourly_forecast():
    ## run base dataset
    df1 = make_empty_df()

    ## zones
    zones_dict = zone_mapping()
    zones = list(zones_dict.keys())
    zones.sort()

    ## create full results dataset
    full_forecast = pd.DataFrame(columns = ['zone', 'date', 'hour', 'mw', 'temp'])
    for z in zones:
        result_df, x = create_analysis_df(z, zones_dict, df1)
        model2 = xgb.XGBRegressor()
        model2.load_model("models/xgb_mod_" + z + ".txt")


        result_df['mw'] = model2.predict(x)
        full_forecast = pd.concat([full_forecast, result_df], axis = 0)
    return full_forecast

def output_data():
    ## run models to get hourly predictions
    hourly_forecast, historical_df = get_hourly_forecast()
    
    ## bring in zone data for the historical peak data
    zzz = pd.read_csv('models/zone_mapping_hist_peak.csv')
    zzz = zzz[['zone', 'hist_peak_mw']].copy()
    
    ## aggregate up to the system to get the peak hour for the day
    system = hourly_forecast.groupby(['hour'], as_index = False).agg({'mw': 'sum'})
    peak_hr = system['hour'].loc[system['mw'] == max(system['mw'])].iloc[0]
    system['peak_hour'] = peak_hr
    
    ## create the peak hour data
    peak_forecast = hourly_forecast.loc[hourly_forecast['hour'] == peak_hr]
    peak_forecast = peak_forecast[['zone', 'date', 'hour', 'mw']].copy()
    peak_forecast.columns = ['zone', 'date', 'peak_hour', 'peak_hour_mw']
    peak_forecast = peak_forecast.merge(zzz, how = 'left', on = 'zone')
    peak_forecast['pct_hist_peak_mw'] = peak_forecast['peak_hour_mw']/peak_forecast['hist_peak_mw']

    ## create the hourly forecast data
    hourly_forecast = hourly_forecast[['zone', 'hour', 'mw', 'temp']]
    hourly_forecast.columns = ['zone', 'hour', 'hourly_mw', 'hourly_temp']
    
    ## finalize system hourly data
    system.columns = ['hour', 'system_mw', 'peak_hour']
    
    ## return 3 datasets
    return peak_forecast, hourly_forecast, system, historical_df

peak_forecast, hourly_forecast, system, historical_df = output_data()

## export each to csv (or change this if proceeding with live solution)
peak_forecast.to_csv('output/peak_hour_forecast.csv', index = False)
hourly_forecast.to_csv('output/hourly_forecast.csv', index = False)
system.to_csv('output/system_hourly_forecast.csv', index = False)
historical_df.to_csv('output/historical_data.csv', mode='a', header=False, index=False)
