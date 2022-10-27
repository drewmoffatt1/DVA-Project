from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime, timedelta
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

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


dfm = px.data.gapminder().query("year==2007")
fig_map = px.scatter_geo(dfm, locations="iso_alpha", color="continent",
                     hover_name="country", size="pop",
                     projection="natural earth")


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
                dbc.Col([html.Div([
                    dcc.Graph(id="load", style={"height": "30%"})
                ])], width=5),
                dbc.Col([html.Div([
                    dcc.Graph(id="temp", style={"height": "30%"})
                ])], width=5)
            ]),
            dbc.Row(
                html.Div([
                   #dcc.Graph(figure=fig_map)
                    dl.Map(children=[dl.TileLayer(url="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                                     dl.LocateControl()],
                           style={'width': "100%", 'height': "100%"}, center=[55.5, 10.5], zoom=8),
                ], style={'width': '1000px', 'height': '500px'})
            )
            ## plot choropleth map
            # dbc.Row(html.Div([
            #     dcc.Graph(id="map")
            # ]))
        ], width=True)
    ])
])


@app.callback(
    Output("load", "figure"),
    Input("Zone", "value")
)
def update_load(zone):
    df1 = update_data(zone)
    fig = px.line(df1, "dt", "mw", markers=True)
    return fig

@app.callback(
    Output("temp", "figure"),
    Input("Zone", "value")
)
def update_temp(zone):
    df1 = update_data(zone)
    fig = px.line(df1, "dt", "temp", markers=True)
    return fig



# @app.callback(
#     Output("map", "figure"),
#     Input("Zone", "value")
# )
# def update_map(zone):
#     fig = dl.TileLayer()
#     return fig

if __name__ == '__main__':
    app.run_server(debug=True)