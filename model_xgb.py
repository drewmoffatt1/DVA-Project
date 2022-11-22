import pandas as pd
import numpy as np
import requests
import json
import datetime as dt
import xgboost as xgb
from pandas.tseries.holiday import get_calendar, nearest_workday, Holiday


class ModelXGB:
    def __init__(self, zone):
        self.zone = zone
        zm = pd.read_csv('Data/zone_mapping_hist_peak.csv')
        zone_dict = dict()
        for i in zm['zone']:
            this_val = dict()
            this_val['lat'] = zm['lat'].loc[zm['zone'] == i].iloc[0]
            this_val['long'] = zm['long'].loc[zm['zone'] == i].iloc[0]
            zone_dict[i] = this_val
        self.zones_dict = zone_dict

    def holiday_cal(self, strdate):
        usfh = get_calendar("USFederalHolidayCalendar")
        juneteenth = Holiday(
            "Juneteenth", month=6, day=19, start_date='2021-10-01', observance=nearest_workday
        )
        if not any(h.name == "Juneteenth" for h in usfh.rules):
            usfh.rules.append(juneteenth)
        return usfh.holidays(strdate, strdate)


    def cycle_encode(self, df, var, minval, maxval):
        space = 2*np.pi/(maxval-minval)
        return np.cos(space*df[var]), np.sin(space*df[var])


    def get_dow_encode(self, row):
        dow_dict = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
        return dow_dict.get(row['dow'])


    def get_season_encode(self, row):
        if row['month'] in [6, 7, 8, 9]:
            return 1
        elif row['month'] in [5, 10]:
            return 2
        elif row['month'] in [4, 11]:
            return 3
        else:
            return 4


    def make_empty_df(self, strdate):
        ## bring in date
        #strdate = get_date()

        ## bring in holidays
        holidays = self.holiday_cal(strdate)

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
        df['hour_cos'], df['hour_sin'] = self.cycle_encode(df, 'hour', 0, 23)
        df['month_cos'], df['month_sin'] = self.cycle_encode(df, 'month', 1, 12)

        ## create dow encoding
        df['dow_num'] = df.apply(self.get_dow_encode, axis = 1)

        ##holiday
        df['holiday'] = df['datetime'].dt.strftime('%Y-%m-%d').isin(holidays)

        ## season encoding
        df['season_num'] = df.apply(self.get_season_encode, axis=1)

        return df[['datetime', 'date', 'hour', 'year', 'dow', 'is_dst', 'is_sah',
           'precovid', 'postcovid', 'weekend', 'recent', 'hour_cos', 'hour_sin',
           'month_cos', 'month_sin', 'dow_num', 'holiday', 'season_num']].copy()


    def api_weather_forecast(self, strdate):
        ## Set parameters for API call
        latitude = str(self.zones_dict.get(self.zone).get('lat'))
        longitude = str(self.zones_dict.get(self.zone).get('long'))
        #strdate = get_date()
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
                                '&end_date=' + strdate
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



    def create_analysis_df(self, base_df, strdate):
        df2 = self.api_weather_forecast(strdate)
        df8 = base_df.merge(df2, on='datetime', how='left')
        result_df = df8[['date', 'hour']].copy()
        result_df['zone'] = self.zone
        x = df8[['year', 'weekend', 'holiday', 'is_dst', 'is_sah', 'precovid',
                 'postcovid', 'temp', 'precip', 'rh', 'pressure', 'windspeed', 'rain', 'snow',
                 'recent', 'hour_cos', 'hour_sin', 'month_cos', 'month_sin', 'dow_num', 'season_num']].copy()
        x['rain'] = x['rain'].astype('bool')
        x['snow'] = x['snow'].astype('bool')
        return result_df, x


    def predict(self, strdate):
        ## run base dataset
        df1 = self.make_empty_df(strdate)

        result_df, x = self.create_analysis_df(df1, strdate)
        model2 = xgb.XGBRegressor()
        model2.load_model("models/xgb_mod_" + self.zone + ".txt")
        result_df['mw'] = model2.predict(x)
        result_df = pd.concat([result_df, x], axis=1)
        return result_df


# m2 = ModelXGB('AEP')
# m2.predict('2022-10-08')
