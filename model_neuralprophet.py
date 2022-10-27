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
from pandarallel import pandarallel


## AR-Net
def train_model(df1):
    df1s = df1[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
    df1s.loc[:, "rain"] = df1s["rain"].astype(int)
    df1s.loc[:, "snow"] = df1s["snow"].astype(int)

    model = NeuralProphet(
        n_forecasts=24,
        n_lags=7*24,
        num_hidden_layers=1,
        d_hidden=16,
        learning_rate=0.01,
        drop_missing=True
    )

    model.add_lagged_regressor(names = list(df1s)[2:])
    df_train, df_test = model.split_df(df1s, freq="H", valid_p=0.10)
    metrics = model.fit(df_train, freq="H", validation_df=df_test)

    ## save results
    torch.save(model.model.state_dict(), 'checkpoints/np_lag24_AR_weights'+df1.zone.values[0]+'.pth')
    metrics['zone'] = df1.zone.values[0]
    metrics.to_csv("checkpoints/metrics_"+df1.zone.values[0]+'.csv')
    return metrics


if __name__ == "__main__":
    set_random_seed(0)

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("Data/03_full_dataset_clean.csv", header=True, inferSchema=True)
    df = df.withColumnRenamed("dt", "ds").withColumnRenamed("mw", "y")
    # imputation
    cols = ["precip", "rh", "pressure", "windspeed"]
    imputer = Imputer(strategy="mean", inputCols=cols, outputCols=cols)
    df = imputer.fit(df).transform(df)
    DF = df.toPandas()

    pandarallel.initialize(nb_workers = 21, progress_bar=True)
    res = DF.groupby("zone").apply(train_model)
    
    # ## parallel by pyspark
    # res = df.groupby("zone").applyInPandas(
    #     train_model,
    #     schema="SmoothL1Loss double, MAE double, RMSE double, Loss double, RegLoss double, SmoothL1Loss_val double, MAE_val double, RMSE_val double, zone string"
    # ).toPandas()

    res.to_csv("checkpoints/np_metrics_results.csv")
