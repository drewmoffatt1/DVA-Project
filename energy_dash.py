from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime, timedelta
import dash_leaflet as dl
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("Data/03_full_dataset_clean.csv", header=True, inferSchema=True)
## filter zone and date
today = datetime.today()
window = 7
zones = df.select("zone").distinct().toPandas().zone.values

def update_data(zone):
    today = datetime(2022, 9, 1)
    date_ext = today - timedelta(window)
    df1 = df.filter((col("zone") == zone) & (col("dt") > date_ext) & (col("dt") <= today)).toPandas()
    return df1

def get_info(zone=None):
    return [html.P(zone)]

info = html.Div(children=get_info(), id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})

tab1 = dbc.Card(
    dbc.CardBody([
        html.Div([dcc.Graph(id="load")])
    ]),
    className="mt-3"
)
tab2 = dbc.Card(
    dbc.CardBody([
        html.Div([dcc.Graph(id="temp")])
    ]),
    className="mt-3"
)

app.layout = dbc.Container([
    dbc.Row([html.H1(children="Energy Load prediction"),
             html.Hr()]),
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Select zone:"),
            dcc.Dropdown(
            zones,
            zones[0],
            id="Zone"
            ),
            html.H4("Select time range:"),
        ]), width=2),
        dbc.Col([
            ## plot energy/temp curve
            dbc.Row([
                dbc.Tabs([
                    dbc.Tab(tab1, label="Load"),
                    dbc.Tab(tab2, label="Temp")
                ])
            ]),
            dbc.Row(
                html.Div([
                   #dcc.Graph(figure=fig_map)
                    dl.Map(children=[dl.TileLayer(), info],center=[39, -98], zoom=4,
                           style={'width': "100%", 'height': "100%"}),
                ], style={'width': '1000px', 'height': '500px'})
            )
        ], width=True)
    ]),
    dbc.Row(html.Br())
])

@app.callback(
    Output("info", "children"),
    Input("Zone", "value")
)
def update_map(zone):
    return get_info(zone)

@app.callback(
    Output("load", "figure"),
    Input("Zone", "value")
)
def update_load(zone):
    df1 = update_data(zone)
    fig = px.line(df1, "dt", "mw", markers=True, width=950, height=300, template="plotly_white")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

@app.callback(
    Output("temp", "figure"),
    Input("Zone", "value")
)
def update_temp(zone):
    df1 = update_data(zone)
    fig = px.line(df1, "dt", "temp", markers=True, width=950, height=300, template="plotly_white")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)