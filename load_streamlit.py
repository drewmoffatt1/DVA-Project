import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors
from model_np import ModelNP
from model_xgb import ModelXGB
from datetime import timedelta
import datetime
from pytz import timezone
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from folium.features import DivIcon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from branca.element import Template, MacroElement

template = """
{% macro html(this, kwargs) %}
<html lang="en">
<body>
<div class='my-legend'>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li>&nbsp<span style='background:#008000;'></span></li>
    <li>50<span style='background:#FF7F50;'></span></li>
    <li>&nbsp<span style='background:#FF7F50;'></span></li>
    <li>&nbsp&nbsp&nbsp&nbsp15,000<span style='background:#FF7F50;'></span></li>
  </ul>
</div>
<br>
<div class='legend-title'>Node size by peak load</div>
</div>

</body>
</html>
<style>
    .my-legend {
        position: relative;
        z-index:9999;
        right: -1155px;
        bottom: 450px;
    }
  .my-legend .legend-title {
    text-align: left;
    margin-top: 8px;
    font-size: 75%;
    }
  .my-legend .legend-scale ul {
    margin: 0;
    padding: 0;
    float: left;
    list-style: none;
    height: 20px;
    }
  .my-legend .legend-scale ul li {
    display: block;
    float: left;
    width: 20px;
    margin-bottom: 6px;
    text-align: left;
    font-size: 80%;
    list-style: none;
    }
  .my-legend ul.legend-labels li span {
    display: block;
    float: left;
    height: 12px;
    width: 50px;
    }
</style>
{% endmacro %}
"""

st.set_page_config(page_title="PJM Day-Ahead Forecasting Tool", layout="wide")
st.markdown(f'''
<style>
.css-af4qln.e16nr0p31 {{margin-top: -75px}}
.main .block-container {{padding-top: 2rem; padding-bottom: 0rem}}
</style>
''', unsafe_allow_html=True)

## load data
@st.cache
def load_history(file):
    hrl = pd.read_csv(file)
    hrl['ds'] = pd.to_datetime(hrl['datetime_beginning_ept'])
    hrl = hrl.groupby(['ds', 'zone']).agg({'mw': 'sum'}).reset_index()

    hrl = hrl[hrl['zone'].isin(zmap['zone'])]
    return hrl

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def forecasting(zones, hrl=None, date=None, model='XGBoost'):
    preds = []

    my_bar = st.sidebar.progress(0)
    pct = 0
    for zone in zones:
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

        pct += 1/len(zones)
        if pct >= 1:
            pct = 1
        my_bar.progress(pct)
    pred_subset = pd.concat(preds)

    return pred_subset


## add geo
def time_select(hrl, date_s, hour):
    hist_h = hrl[(hrl['ds'].dt.date == date_s) & (hrl['ds'].dt.hour == int(hour))]
    return hist_h.merge(zmap, on='zone')

@st.cache
def load_data():
    zmap = pd.read_csv('Data/zone_mapping_hist_peak.csv')
    zones = zmap['zone'].unique()
    #peak = pd.read_csv('Data/peak_hour_forecast.csv')[['zone', 'hist_peak_mw']]
    return zmap, zones

zmap, zones = load_data()

###########
## side bar
st.sidebar.header('PJM Day-Ahead Forecasting Tool')
# model = st.sidebar.selectbox('Choose model:',
#                              ('XGBoost', 'neuralprophet'),
#                              index=0,
#                              help='XGBoost model is based on weather forecast. '
#                                   'NeuralProphet model based on both weather forcast and 7-days historical records')
model = 'XGBoost'
tz = timezone('EST')
date_d = datetime.datetime.now(tz).date() + timedelta(1)

if model=='neuralprophet':
    uploaded_file = st.sidebar.file_uploader("Upload metered hourly load (at least 7 days)")
    st.sidebar.caption(
        "(Note: PJM metered load can be downloaded from: https://dataminer2.pjm.com/feed/hrl_load_metered. Data from 2022/11/01 to 2022/11/07 are preloaded for demo.)")

    if uploaded_file is not None:
        hrl = load_history(uploaded_file)
    else:
        hrl = load_history('Data/hrl_load_metered_7.csv')

    start_date = hrl['ds'].max().date() + timedelta(1)
    end_date = hrl['ds'].max().date() + timedelta(1)
    date_s = st.sidebar.date_input('Select Date:', value=end_date, min_value=start_date, max_value=end_date,
                                   help='Model will forcast the 24h Load after the given 7-days.')
    pred_all = forecasting(zones, hrl=hrl, model='neuralprophet')
    pred_all = pred_all.rename(columns={'y': 'mw'})
    pred_all['hour'] = pred_all['ds'].dt.hour
    pred_all = pred_all[pred_all['ds'].dt.date == date_s]
    #pred_all = pred_all.set_index('hour').reset_index()
    pred_all.index = pred_all['hour'].values
elif model=='XGBoost':
    date_s = st.sidebar.date_input('Select Date:', value=date_d,
                                   help='Recent day if the weather forecast is available.')
    try:
        pred_all = forecasting(zones, date=date_s, model='XGBoost')
    except:
        st.sidebar.exception('open-meteo.com connection timeout')
        pred_all = pd.read_csv('Data/pred_all_xgb.csv', index_col=0)
        date_s = pred_all['date'].values[0]
else:
    st.error('Not implemented!')


container = st.sidebar.container()
if 'all_value' not in st.session_state.keys():
    st.session_state.all_value = False

All = st.sidebar.checkbox("Select all", value=st.session_state.all_value)
if All:
    zone_s = container.multiselect("Select zones:", zmap.zone.unique(), default=zmap.zone.unique(), key='zone_k',
                                   help='Multiple selection for transmission zones to investigate. '
                                        'Selected zones will be highlighted on map and plots.'
                                        'Zones can also be selected by clicking the markers on the map.')
    if 'zone_m' in st.session_state.keys():
        st.session_state.zone_m = zone_s
    if len(zone_s) == 21:
        st.session_state.all_value = True
else:
    if 'zone_d' not in st.session_state.keys():
        st.session_state.zone_d = []
    elif len(st.session_state.zone_d) == len(zones):
        st.session_state.zone_d = []
    zone_s = container.multiselect("Select one or more options:", zmap.zone.unique(), default=st.session_state.zone_d,
                                   help='Multiple selection for transmission zones to investigate. '
                                        'Selected zones will be highlighted on map and plots. '
                                        'Zones can also be selected by clicking the markers on the map.')
    #st.write(st.session_state.zone_d)
# zone_s = st.sidebar.multiselect('Zones:', zmap.zone.unique(), default=['AEP', 'CE'], key='zone_k')
# if len(zone_s) == 0:
#     zone_s = zones.tolist()

# if st.session_state.all_value and (len(st.session_state.zone_d) != 21):
#     st.experimental_rerun()

#st.write(st.session_state)
hour = st.sidebar.slider('Select Hour:', min_value=0, max_value=23, value=int(datetime.datetime.now().strftime("%H")),
                         help='The realtime load will shown on the map and the trend on the cards.')
time_s = datetime.datetime(date_s.year, date_s.month, date_s.day, int(hour), 0)

def update_by_zone_date(pred_all, zone_s, date_s, hrl=None):
    ## model prediction
    if model == 'neuralprophet':
        # pred_subset = forecasting(zone_s, hrl=hrl, model='neuralprophet')
        # pred_subset = pred_subset.rename(columns={'y': 'mw'})
        pred_subset = pred_all[pred_all['zone'].isin(zone_s)]
        # hrl = pd.concat([hrl, pred_subset[['ds', 'zone', 'mw']]], axis=0)
        # data for map
        hist_h = time_select(pred_all, date_s, hour)
    elif model == 'XGBoost':
        pred_all['ds'] = pd.to_datetime(pred_all['date']) + pred_all['hour'].astype('timedelta64[h]')
        pred_subset = pred_all[pred_all['zone'].isin(zone_s)]
        #pred_subset['source'] = 'pred'
        # data for map
        hist_h = time_select(pred_all, date_s, hour)[['zone', 'mw', 'temp', 'lat', 'long', 'full_zone_name', 'hist_peak_mw']]
    else:
        st.error('Not implemented!')
    # st.experimental_show(pred_subset)
    pred_max = pred_all.groupby('zone').agg({'mw': max}).reset_index()
    pred_max = pred_max.rename(columns={'mw': 'mw_max'})
    pred_max_h = pred_all[['zone', 'mw']].groupby('zone').idxmax().reset_index()
    pred_max_h = pred_max_h.rename(columns={'mw': 'h_max'})
    hist_h = hist_h.merge(pred_max, on='zone')
    hist_h = hist_h.merge(pred_max_h, on='zone')

    # st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    # hist_h = z_selected(hist_h, zone_s)
    # hist_h = scale_color(hist_h)
    hist_h['mw'] = round(hist_h['mw'], 2)
    hist_h['pct'] = hist_h['mw_max'] / hist_h['hist_peak_mw']

    return pred_subset, hist_h

# st.sidebar.markdown("<hr>", unsafe_allow_html=True)
## status
if model == 'XGBoost':
    pred_subset, hist_h = update_by_zone_date(pred_all, zone_s, date_s, hrl=None)
elif model == 'neuralprophet':
    pred_subset, hist_h = update_by_zone_date(pred_all, zone_s, date_s, hrl=hrl)

if pred_all.shape[0] > 0:
    st.sidebar.success('Forecasting on all zones finished!', icon="✅")


@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(pred_all[['ds', 'zone', 'temp', 'mw']])
st.sidebar.download_button(
    "Download forecasts",
    csv,
    "forecast_{}.csv".format(date_s.strftime('%Y-%m-%d')),
    "text/csv",
    key='download-csv',
    help='To download forcasts in csv format for all zones.'
)

#########
## main
def linear_scale(x):
    sc = (x - min(x)) / (max(x) - min(x))
    return sc * 40 + 10

rad = linear_scale(hist_h['mw_max']).values

def color_map(hist_h, i):
    step = cm.LinearColormap(['green', 'yellow', 'red'],
                            vmin=0, vmax=1,
                            caption='step')
    return step(hist_h.loc[i, 'pct'])

agg_mw = pred_all.groupby('hour').agg({'mw': sum})
agg_h = agg_mw['mw'].argmax()
mw_sys = pred_all.loc[pred_all['hour'] == agg_h, ['zone', 'mw']]
mw_sys = mw_sys.rename(columns={'mw': 'mw_sys'})
hist_h = hist_h.merge(mw_sys, on='zone')

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric('System Peak Load', f'{round(max(agg_mw["mw"])):,}'+' mw')
with mcol2:
    st.metric('System Peak Hour', str(agg_h) + 'H')
if (len(zone_s) < len(zones)) and (len(zone_s) > 0):
    agg_s = pred_subset.groupby('hour').agg({'mw': sum})
    agg_sh = agg_s['mw'].argmax()
    agg_sm = agg_s['mw'].max()
    with mcol3:
        st.metric('Peak Load for Selected Zones', f'{(round(agg_sm)):,}'+' mw')
    with mcol4:
        st.metric('Peak Hour for Selected Zones', str(agg_sh) + 'H')
# st.markdown('System Peak Load: **{}** mw<br>System Peak Hour: **{}H**'.format(
#     str(round(max(agg_mw['mw']), 2)),
#     #date_s.strftime('%Y-%m-%d'),
#     str(agg_h)
# ), unsafe_allow_html=True)
top5 = hist_h.nlargest(5, 'mw_max')
m = folium.Map([40, -80], zoom_start=6)
locations = list(zip(hist_h['lat'], hist_h['long']))
for i in range(len(zones)):
    if zones[i] in zone_s:
        b_color = 'black'
    else:
        b_color = color_map(hist_h, i)
    folium.CircleMarker(location=locations[i], radius=int(rad[i]), fill=True,
                        color=b_color, fill_color=color_map(hist_h, i), fill_opacity=0.5,
                        tooltip="<b>{}({})</b><br>Peak Load ({}H): {}<br>Load when system peak ({}H): {}<br>% of historical peak: {}%<br>Realtime Load ({}H): {}<br>Temperature ({}H): {}".format(
                            hist_h.loc[i, 'full_zone_name'],
                            hist_h.loc[i, 'zone'],
                            str(hist_h.loc[i, 'h_max']),
                            f'{round(hist_h.loc[i, "mw_max"]):,}',
                            str(agg_h),
                            f'{round(hist_h.loc[i, "mw_sys"]):,}',
                            round(hist_h.loc[i, 'mw_max']/hist_h.loc[i, 'hist_peak_mw']*100, 2),
                            str(hour),
                            f'{round(hist_h.loc[i, "mw"]):,}',
                            str(hour),
                            str(hist_h.loc[i, 'temp'])
    )).add_to(m)

    if zones[i] in top5['zone'].values:
        folium.map.Marker(
            location=locations[i],
            icon=DivIcon(
                #icon_size=(250,36),
                icon_anchor=(5,0),
                html='<div style="font-size: 10pt">{} {}</div>'.format(zones[i], f'{round(hist_h.loc[i, "mw_max"]):,}'),
                )
            ).add_to(m)

colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1, caption='Color: % of Historical Peak')
colormap.width = 250
colormap.add_to(m)

# cm2 = cm.StepColormap(['black'], vmin=20, vmax=100, caption='50----------------------15000mw')
# cm2.width = 200
# m.add_child(cm2)

macro = MacroElement()
macro._template = Template(template)
m.get_root().add_child(macro)

click_out = st_folium(m, width=1300, height=500, returned_objects=['last_object_clicked'])


# if not All:
#     st.session_state.zone_m = zone_s.copy()
if click_out['last_object_clicked'] is None:
    st.session_state.zone_m = zone_s.copy()
else:
    zidx = (abs(zmap['lat'] - click_out['last_object_clicked']['lat']) + abs(zmap['long'] - click_out['last_object_clicked']['lng'])).argmin()
    zone_co = zmap.loc[zidx, 'zone']

    if not zone_co in st.session_state.zone_m:
        st.session_state.zone_m.append(zone_co)
        zone_s = st.session_state.zone_m.copy()
        #st.session_state.zone_d.append(zone_co)
    else:
        st.session_state.zone_m.remove(zone_co)
        zone_s = st.session_state.zone_m.copy()
    st.session_state.zone_d = zone_s
    if len(zone_s) != len(zones):
        st.session_state.all_value = False
    elif len(st.session_state.zone_d) != len(zones):
        st.session_state.all_value = False
    else:
        st.session_state.all_value = True
    st.experimental_rerun()
    pred_subset, hist_h = update_by_zone_date(zone_s, date_s, hrl=None)

# if len(zone_s) != len(zones):
#     st.session_state.all_value = False
# else:
#     st.session_state.all_value = True
pred_all['selected'] = pred_all['zone'].isin(zone_s).astype('str')

col1, col2, col3 = st.columns([4, 3, 1])
with col1:
    st.write('System Load Forecast (' + time_s.strftime('%Y-%m-%d')+')')
    try:
        fig1 = px.area(pred_all, x='ds', y='mw', color='zone',
                       template="plotly_white", symbol='selected',
                       symbol_map={'True': 'circle', 'False': 'line-ew'},
                       height=200)
    except Exception:
        fig1 = px.area(pred_all, x='ds', y='mw', color='zone',
                       template="plotly_white", symbol='selected',
                       symbol_map={'True': 'circle', 'False': 'line-ew'},
                       height=200)
    fig1.update_layout(margin={"t": 0, "b": 0, "l": 0, "r": 0},
                       xaxis_title=None, yaxis_title="Hourly Load (mw)")
                       #legend=dict(orientation="h"))
    fig1.update_yaxes(gridcolor='lightgrey')
    # if model == 'neuralprophet':
    #     pred_date = end_date + timedelta(1)
    #     fig1.add_vline(x=datetime.datetime(pred_date.year, pred_date.month, pred_date.day, 0, 0), line_dash="dash",
    #                    line_width=1, line_color='red')
    fig1.add_vline(x=time_s, line_dash="dash")

    st.plotly_chart(fig1, use_container_width=True)

with col2:
    if len(zone_s) > 0:
        SA = 'Selected'
        agg_subset = pred_subset.groupby('hour').agg({'mw': 'sum'}).reset_index()
        agg_temp = pred_subset.groupby('hour').agg({'temp': 'mean'}).reset_index()
    else:
        SA = 'All'
        agg_subset = pred_all.groupby('hour').agg({'mw': 'sum'}).reset_index()
        agg_temp = pred_all.groupby('hour').agg({'temp': 'mean'}).reset_index()

    st.write('{} zone(s) Load / Average Temperature'.format(SA))
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    fig2.add_trace(
        go.Scatter(x=agg_subset['hour'], y=agg_subset['mw'], name="{} zones Load (mw)".format(SA),
                   marker=dict(symbol=['circle']*24)),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(x=agg_temp['hour'], y=round(agg_temp['temp'],2), name="Temperature (°F)"),
        secondary_y=True,
    )
    fig2.update_layout(
        height=200,
        margin={"t": 0, "b": 0, "l": 0, "r": 0},
        xaxis_title=None,
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    mw1 = agg_mw.loc[agg_mw.index==hour, 'mw'].values[0]
    if hour >= 1:
        mw0 = agg_mw.loc[agg_mw.index == (hour-1), 'mw'].values[0]
        delta_mw = str(round(mw1-mw0))
    else:
        delta_mw = ''
    st.metric('System Load ('+str(hour)+'H)', f'{(round(mw1)):,}', delta_mw)

    if len(zone_s) > 0:
        agg_mw_s = pred_subset.groupby('hour').agg({'mw': sum})
        mw1 = agg_mw_s.loc[agg_mw_s.index == hour, 'mw'].values[0]
        if hour >= 1:
            mw0 = agg_mw_s.loc[agg_mw_s.index == (hour - 1), 'mw'].values[0]
            delta_mw = str(round(mw1-mw0))
        else:
            delta_mw = ''
        st.metric('Selected Load ('+ str(hour)+'H)', f'{(round(mw1)):,}', delta_mw)

#st.write(st.session_state)