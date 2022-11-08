import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.figure_factory as ff
import plotly.express as px
import plotly.colors
from streamlit_card import card

st.set_page_config(layout="wide")

## load data
@st.experimental_memo
def load_pred():
    pred = pd.read_csv('data/hourly_forecast.csv')
    zmap = pd.read_csv('Data/zone_mapping.csv')
    pred = pred.merge(zmap, on='zone')
    return pred

## filter pred
def filter_pred():
    zone_inputs = st.multiselect('Zones:', pred.zone.unique(), default='AE')
    pred_subset = pred[pred['zone'].isin(zone_inputs)]
    # st.experimental_show(zone_inputs)
    return pred_subset

pred = load_pred()

## load curves
pred_subset = filter_pred()
# st.experimental_show(pred_subset)
fig1 = px.line(pred_subset, x='hour', y='hourly_mw', color='zone',
               markers=True, template="plotly_white", log_y=True)
fig1.update_layout(xaxis={"dtick":1},margin={"t":0,"b":0})
fig1.update_xaxes(showgrid=False, range=[0, 23])
fig1.update_yaxes(gridcolor='grey')
st.plotly_chart(fig1, use_container_width=True)

def scale_color(pred_h):
    YR = plotly.colors.PLOTLY_SCALES["YlOrRd"]
    plotly.colors.find_intermediate_color(YR[0][1], YR[-1][1], 0.5, colortype='rgb')
    peak = pd.read_csv('Data/peak_hour_forecast.csv')
    pred_h = pd.merge(pred_h, peak, on='zone')
    pct = pred_h.hourly_mw.values / pred_h.hist_peak_mw.values
    pred_h['color'] = [plotly.colors.unlabel_rgb(plotly.colors.find_intermediate_color(YR[0][1], YR[-1][1], p, colortype='rgb')) for p in pct]
    return pred_h

hour = st.slider('Select Hour:', min_value=0, max_value=23, value=12)
pred_h = pred[pred.hour == hour]
pred_h = scale_color(pred_h)

#st.experimental_show(pred_h)
## load map
st.pydeck_chart(
    pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=40,
        longitude=-80.00,
        zoom=4,
        pitch=0,
        height=700
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=pred_h,
            pickable=True,
            opacity=0.8,
            radius_scale=6,
            radius_min_pixels=1,
            radius_max_pixels=100,
            get_position='[long, lat]',
            get_fill_color='color',
            get_radius="hourly_mw",
        ),
    ],
    tooltip={"text": "{full_zone_name}\nLoad: {hourly_mw}"}),
    use_container_width=True)