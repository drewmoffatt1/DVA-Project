import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.figure_factory as ff
import plotly.express as px
import plotly.colors
# from utils import weather_forcast, load_model, pjm_load, model_predict
from model_np import ModelNP
from model_xgb import ModelXGB
from datetime import timedelta
import datetime

st.set_page_config(layout="wide")


## load data
@st.experimental_memo
def load_history(file):
    hrl = pd.read_csv(file)
    hrl['ds'] = pd.to_datetime(hrl['datetime_beginning_ept'])
    hrl = hrl.groupby(['ds', 'zone']).agg({'mw': 'sum'}).reset_index()

    hrl = hrl[hrl['zone'].isin(zmap['zone'])]
    return hrl


@st.experimental_memo
def filter_pred(zone_s, hrl=None, date=None, model='neuralprophet'):
    preds = []
    for zone in zone_s:
        if model == 'neuralprophet':
            m = ModelNP(zone)
            p = m.predict(hrl)
        elif model == 'XGBoost':
            m = ModelXGB(zone)
            p = m.predict(date.strftime('%Y-%m-%d'))
        else:
            return 'Not implemented'
        p['zone'] = zone
        preds.append(p)
    pred_subset = pd.concat(preds)

    return pred_subset, zone_s


## add geo
def time_select(hrl, date_s, hour):
    hist_h = hrl[(hrl['ds'].dt.date == date_s) & (hrl['ds'].dt.hour == int(hour))]
    return hist_h.merge(zmap, on='zone')


## scale color by load
def scale_color(hist_h):
    Reds = plotly.colors.PLOTLY_SCALES["RdBu"]
    # plotly.colors.find_intermediate_color(YR[0][1], YR[-1][1], 0.5, colortype='rgb')

    # hist_h = pd.merge(hist_h, zmap, on='zone')
    pct = hist_h['mw'].values / hist_h['hist_peak_mw'].values
    hist_h['color'] = [
        plotly.colors.unlabel_rgb(plotly.colors.find_intermediate_color(Reds[0][1], Reds[-1][1], p, colortype='rgb'))
        for p in pct]
    return hist_h


## highlight selected
def z_selected(hist_h, zone_s):
    selected = []
    for z in hist_h['zone']:
        if z in zone_s:
            selected.append([255, 127, 0])
        else:
            selected.append([255, 255, 255])
    hist_h['selected'] = selected
    return hist_h


@st.experimental_memo
def load_data():
    zmap = pd.read_csv('Data/zone_mapping_hist_peak.csv')
    #peak = pd.read_csv('Data/peak_hour_forecast.csv')[['zone', 'hist_peak_mw']]
    return zmap


zmap = load_data()

###########
## side bar
model = st.sidebar.selectbox('Choose model:',
                             ('XGBoost', 'neuralprophet', 'ANN', 'RandomForest', 'TransferFunction'),
                             index=1)

if model=='neuralprophet':
    uploaded_file = st.sidebar.file_uploader("Upload metered hourly load (at least 7 days)")
    st.sidebar.caption(
        "(Note: PJM metered load can be downloaded from: https://dataminer2.pjm.com/feed/hrl_load_metered. Data from 2022/11/01 to 2022/11/07 are preloaded for demo.)")

    if uploaded_file is not None:
        hrl = load_history(uploaded_file)
    else:
        hrl = load_history('Data/hrl_load_metered_7.csv')

    start_date = hrl['ds'].min().date()
    end_date = hrl['ds'].max().date()
    date_s = st.sidebar.date_input('Select Date:', value=end_date, min_value=start_date, max_value=end_date + timedelta(1))
elif model=='XGBoost':
    date_s = st.sidebar.date_input('Select Date:', value=datetime.date.today())
else:
    st.error('Not implemented!')

zone_s = st.sidebar.multiselect('Zones:', zmap.zone.unique(), default='AEP')
if len(zone_s) == 0:
    zone_s = ['AEP']


hour = st.sidebar.slider('Select Hour:', min_value=0, max_value=23, value=12)

## model prediction
if model == 'neuralprophet':
    pred_subset, zone_s = filter_pred(zone_s, hrl=hrl, model='neuralprophet')
    pred_subset = pred_subset.rename(columns={'y': 'mw'})
    hrl = pd.concat([hrl, pred_subset[['ds', 'zone', 'mw']]], axis=0)
    # data for map
    hist_h = time_select(hrl, date_s, hour)
elif model == 'XGBoost':
    pred_subset, zone_s = filter_pred(zone_s, date=date_s, model='XGBoost')
    pred_subset['ds'] = pd.to_datetime(pred_subset['date']) + pred_subset['hour'].astype('timedelta64[h]')
    pred_subset['source'] = 'pred'
    # data for map
    hist_h = time_select(pred_subset, date_s, hour)[['zone', 'mw', 'lat', 'long', 'full_zone_name', 'hist_peak_mw']]
else:
    st.error('Not implemented!')
# st.experimental_show(pred_subset)

# st.sidebar.markdown("<hr>", unsafe_allow_html=True)
hist_h = z_selected(hist_h, zone_s)
hist_h = scale_color(hist_h)
time_s = datetime.datetime(date_s.year, date_s.month, date_s.day, int(hour), 0)

# st.sidebar.markdown("<hr>", unsafe_allow_html=True)
## status
if pred_subset.shape[0] > 0:
    st.sidebar.success('Prediction on ' + zone_s[-1] + ' finished!', icon="✅")

#########
## main
col1, col2 = st.columns(2)
with col1:
    st.write('Load Forecast')
    try:
        fig1 = px.line(pred_subset, x='ds', y='mw', color='zone', symbol='source',
                       symbol_map={"PJM": "circle", "pred": "x"},
                       markers=True, template="plotly_white", log_y=True,
                       height=200)
    except Exception:
        fig1 = px.line(pred_subset, x='ds', y='mw', color='zone', symbol='source',
                       symbol_map={"PJM": "circle", "pred": "x"},
                       markers=True, template="plotly_white", log_y=True,
                       height=200)
    fig1.update_layout(margin={"t": 0, "b": 0, "l": 0, "r": 0}, hovermode="x unified",
                       xaxis_title=None, yaxis_title="Load")
    fig1.update_yaxes(gridcolor='lightgrey')
    if model == 'neuralprophet':
        pred_date = end_date + timedelta(1)
        fig1.add_vline(x=datetime.datetime(pred_date.year, pred_date.month, pred_date.day, 0, 0), line_dash="dash",
                       line_width=1, line_color='red')
    fig1.add_vline(x=time_s, line_dash="dash")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.write('Weather Forecast')
    fig2 = px.line(pred_subset, x='ds', y='temp', color='zone',
                   markers=True, template="plotly_white", log_y=True,
                   hover_data=['rh', 'precip', 'pressure', 'windspeed', 'rain', 'snow'],
                   height=200)
    fig2.update_layout(margin={"t": 0, "b": 0, "l": 0, "r": 0},
                       xaxis_title=None, yaxis_title="temperature")
    fig2.update_yaxes(gridcolor='lightgrey')
    fig2.add_vline(x=time_s, line_dash="dash")
    st.plotly_chart(fig2, use_container_width=True)

## Load map
# st.experimental_show(hist_h)
st.write("Load map (" + time_s.strftime('%Y-%m-%d %H:%M:%S') + ")")
L1 = pdk.Layer(
    'ScatterplotLayer',
    data=hist_h,
    pickable=True,
    stroked=True,
    filled=True,
    opacity=0.8,
    radius_scale=6,
    radius_min_pixels=1,
    radius_max_pixels=100,
    line_width_min_pixels=2,
    get_position='[long, lat]',
    get_fill_color='color',
    get_line_color='selected',
    get_radius="mw",
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=40,
            longitude=-80.00,
            zoom=5,
            pitch=0,
            height=700
        ),
        layers=[L1],
        tooltip={"text": "{full_zone_name}({zone})\nLoad: {mw}"}),
    use_container_width=True)
