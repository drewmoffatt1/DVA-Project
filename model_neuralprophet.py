from neuralprophet import NeuralProphet, set_random_seed
# from model_prophet import load_data##, eval_model
import pandas as pd
import numpy as np
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Imputer
import torch
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse
import matplotlib.pyplot as plt

# df = pd.read_csv("Data/03_full_dataset_clean.csv")
# df = df.rename(columns = {"dt":"ds", "mw":"y"})
# df.ds = pd.to_datetime(df.ds, utc=True)
# df.ds = df.ds.apply(lambda x: str(x)[:-6])
# df.ds = pd.to_datetime(df.ds)
set_random_seed(0)

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("Data/03_full_dataset_clean.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("dt", "ds").withColumnRenamed("mw", "y")
# imputation
cols = ["precip", "rh", "pressure", "windspeed"]
imputer = Imputer(strategy="mean", inputCols=cols, outputCols=cols)
df = imputer.fit(df).transform(df)


## subset
df1 = df.filter(col("zone") == "AE").toPandas()
# df1.loc[df1.ds.isin(df1.ds[df1.ds.duplicated()])]
# df1 = df1.drop_duplicates(subset=['ds'])

df1s = df1[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
df1s.loc[:, "rain"] = df1s["rain"].astype(int)
df1s.loc[:, "snow"] = df1s["snow"].astype(int)

m = NeuralProphet(
    n_forecasts=24,
    n_lags=24,
    learning_rate=0.01,
    drop_missing=True
    )

m.add_lagged_regressor(names = list(df1s)[2:])
m.highlight_nth_step_ahead_of_each_forecast(24)

df_train, df_test = m.split_df(df1s, freq="H", valid_p=0.10)

metrics = m.fit(df1s, freq="H", validation_df=df_test)

torch.save(m.model.state_dict(), 'checkpoints/np_lag24_weights.pth')

param = m.plot_parameters()
param.figure.show()

# df_test.y[24:] = np.nan
pred = m.predict(df_test)
err1 = mape(pred.y[24:], pred.yhat1[24:,])

ax1 = metrics[['SmoothL1Loss-24', 'SmoothL1Loss-24_val']].plot()
ax1.figure.show()
ax2 = pred[["y", 'yhat1']].plot()
ax2.figure.show()

## plot latest 24
ax = m.plot(pred[-7*24:])
ax.figure.show()

future = m.make_future_dataframe(df_train, periods=24)
pred_f = m.predict(future)
forecast = m.predict(future, raw=True, decompose=False)


## AR-Net
model = NeuralProphet(
    n_forecasts=24,
    n_lags=7*24,
    num_hidden_layers=1,
    d_hidden=32,
    learning_rate=0.01,
    drop_missing=True
    )

model.add_lagged_regressor(names = list(df1s)[2:])
df_train, df_test = model.split_df(df1s, freq="H", valid_p=0.10)

metrics2 = model.fit(df_train, freq="H", validation_df=df_test)

torch.save(model.model.state_dict(), 'checkpoints/np_lag24_AR_weights.pth')

ax1 = metrics2[['SmoothL1Loss-24', 'SmoothL1Loss-24_val']].plot()
ax1.figure.show()

## plot latest 24
ax1 = metrics2[['SmoothL1Loss-24', 'SmoothL1Loss-24_val']].plot()
ax1.figure.show()

pred2 = model.predict(df_test)
err2 = mape(pred2.y[48:], pred2.yhat1[48:,])

model.highlight_nth_step_ahead_of_each_forecast(1)
ax = model.plot(pred2[-7*24:])
ax.figure.show()

if __name__ == "__main__":

    load_data()
