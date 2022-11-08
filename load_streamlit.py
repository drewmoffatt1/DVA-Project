import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.figure_factory as ff
import plotly.express as px
import plotly.colors
#from utils import weather_forcast
import datetime

st.set_page_config(layout="wide")

## load data
@st.experimental_memo
def load_pred():
    pred = pd.read_csv('Data/hourly_forecast.csv')
    zmap = pd.read_csv('Data/zone_mapping.csv')
    pred = pred.merge(zmap, on='zone')
    return pred

## filter pred
def filter_pred():
    zone_s = st.sidebar.multiselect('Zones:', pred.zone.unique(), default='AE')
    pred_subset = pred[pred['zone'].isin(zone_s)]
    # st.experimental_show(zone_inputs)
    return pred_subset, zone_s

## scale color by load
def scale_color(pred_h):
    Reds = plotly.colors.PLOTLY_SCALES["RdBu"]
    #plotly.colors.find_intermediate_color(YR[0][1], YR[-1][1], 0.5, colortype='rgb')
    peak = pd.read_csv('Data/peak_hour_forecast.csv')
    pred_h = pd.merge(pred_h, peak, on='zone')
    pct = pred_h.hourly_mw.values / pred_h.hist_peak_mw.values
    pred_h['color'] = [plotly.colors.unlabel_rgb(plotly.colors.find_intermediate_color(Reds[0][1], Reds[-1][1], p, colortype='rgb')) for p in pct]
    return pred_h

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=pred_h.hist_peak_mw.values, clip=True)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd)
    # cl = [mapper.to_rgb(p) for p in pct]

## highlight selected
def z_selected(pred_h, zone_s):
    selected = []
    for z in pred_h['zone']:
        if z in zone_s:
            selected.append([0,0,255])
        else:
            selected.append([255,255,255])
    pred_h['selected'] = selected
    return pred_h

pred = load_pred()

## side bar
pred_subset, zone_s = filter_pred()
# st.experimental_show(pred_subset)
st.sidebar.date_input('Select Date:', value=datetime.date.today())
hour = st.sidebar.slider('Select Hour:', min_value=0, max_value=23, value=12)

## filter by hour
pred_h = pred[pred.hour == hour]
pred_h = z_selected(pred_h, zone_s)
pred_h = scale_color(pred_h)


## curves
col1, col2 = st.columns(2)

with col1:
    st.write('Load Forecast')
    fig1 = px.line(pred_subset, x='hour', y='hourly_mw', color='zone',
                    markers=True, template="plotly_white", log_y=True, 
                    height=200)
    fig1.update_layout(xaxis={"dtick":1},margin={"t":0,"b":0, "l":0, "r":0}, hovermode="x unified")
    fig1.update_xaxes(showgrid=False, range=[0, 23])
    fig1.update_yaxes(gridcolor='lightgrey')
    fig1.add_vline(x=hour, line_dash="dash")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.write('Weather Forecast')
    fig2 = px.line(pred_subset, x='hour', y='temp', color='zone',
                    markers=True, template="plotly_white", log_y=True,
                    hover_data=['rh', 'precip', 'pressure', 'windspeed', 'rain', 'snow'],
                    height=200)
    fig2.update_layout(xaxis={"dtick":1},margin={"t":0,"b":0, "l":0, "r":0})
    fig2.update_xaxes(showgrid=False, range=[0, 23])
    fig2.update_yaxes(gridcolor='lightgrey')
    fig2.add_vline(x=hour, line_dash="dash")
    st.plotly_chart(fig2, use_container_width=True)


## Load map
# st.experimental_show(pred_h)
L1 = pdk.Layer(
            'ScatterplotLayer',
            data=pred_h,
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
            get_radius="hourly_mw",
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
    tooltip={"text": "{full_zone_name}({zone})\nLoad: {hourly_mw}"}),
    use_container_width=True)