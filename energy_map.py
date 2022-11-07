from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime, timedelta
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from neuralprophet import load

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("Data/03_full_dataset_clean.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("dt", "ds").withColumnRenamed("mw", "y")
## filter zone and date
today = datetime.today()
window = 7
zones = df.select("zone").distinct().toPandas().zone.values

def update_data(zone):
    today = datetime(2022, 9, 1)
    date_ext = today - timedelta(window)
    df1 = df.filter((col("zone") == zone) & (col("ds") > date_ext) & (col("ds") <= today)).toPandas()
    return df1


def get_info(zone=None):
    return [html.P(zone)]

info = html.Div(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"})


select_zone = html.Div(
    dcc.Dropdown(zones, zones[0], id="Zone", style={"zIndex":"1000"}),
    style={"position": "absolute", "top": "10px", "left": "50px", "zIndex": "1000", "width": "10%"})


app.layout = dl.Map(id="map", children=[dl.TileLayer(), select_zone, info],
                    center=[39, -98], zoom=5,
                    style={"width":"100vw", "height":"100vh"})

@app.callback(
    Output("info", "children"),
    Input("Zone", "value")
)
def model_pred(zone):
    m = load("checkpoints/np_lag24_AR_weights_"+zone+".pth")
    df0 = update_data(zone)
    #print(df0)
    df1 = df0[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
    df1.loc[:, "rain"] = df1["rain"].astype(int)
    df1.loc[:, "snow"] = df1["snow"].astype(int)
    future = m.make_future_dataframe(df1)
    p1 = m.predict(future, raw=True, decompose=False)
    future.y[-24:] = p1.transpose().iloc[1:, 0].astype("double")
    future["source"] = "JPM"
    future.source[-24:] = "pred"
    #print(future)
    fig = px.line(future, "ds", "y", color="source", markers=True, height=200, template="plotly_white")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return html.Div([dcc.Graph(figure=fig)])


if __name__ == '__main__':
    app.run_server(debug=True)