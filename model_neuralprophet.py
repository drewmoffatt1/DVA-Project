from neuralprophet import NeuralProphet, set_random_seed, save, load
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
import itertools


## AR-Net
def run_model(df1):
    df1s = df1[["ds", "y", "temp", "precip", "rh", "pressure", "windspeed", "rain", "snow"]]
    df1s.loc[:, "rain"] = df1s["rain"].astype(int)
    df1s.loc[:, "snow"] = df1s["snow"].astype(int)

    param_grid = {
        'n_lags': [24, 7*24],
        'd_hidden': [8, 16, 64]
    }

    all_params = [
        dict(zip(param_grid.keys(), v))
        for v in itertools.product(*param_grid.values())
    ]
    m1 = []
    mdlist = []
    mtlist = []
    for params in all_params:
        print(params)
        model, mt = train_model(df1, params)
        ## mape
        pred1 = model.predict(df1s)
        pred1 = pd.merge(df1, pred1, on="ds")
        test1 = pred1.loc[pred1.test==1] ## or df_test
        m1.append(mape(test1.y_x, test1.yhat1))
        print(m1)
        mdlist.append(model)
        mtlist.append(mt)

    tres = pd.DataFrame(all_params)
    tres['mape'] = m1
    tres['zone'] = df1.zone.values[0]
    idx = m1.index(min(m1))

    # save(mdlist[idx], 'checkpoints_f24/np_f24_'+df1.zone.values[0]+'.pth')
    # mtlist[idx].to_csv("checkpoints_f24/metrics_f24_"+df1.zone.values[0]+'.csv')
    # Res = tres.iloc[[idx], ]

    ## train with full dataset
    model, mt = train_model(df1, all_params[idx])    
    save(model, 'checkpoints_f24/np_f24_'+df1.zone.values[0]+'.pth')
    mt.to_csv("checkpoints_f24/metrics_f24_"+df1.zone.values[0]+'.csv')

    Res = tres.iloc[[idx], ]
    
    return Res

def train_model(df1, params):
    df1s = df1[["ds", "y", "temp", "rh", "precip", "pressure", "windspeed", "rain", "snow"]]
    df1s.loc[:, "rain"] = df1s["rain"].astype(int)
    df1s.loc[:, "snow"] = df1s["snow"].astype(int)

    model = NeuralProphet(
        growth="off",
        n_forecasts=24,
        num_hidden_layers=1,
        learning_rate=0.01,
        drop_missing=True,
        **params        
    )

    model.add_lagged_regressor(names = list(df1s)[2:])
    df_train, df_test = model.split_df(df1s, freq="H", valid_p=0.10)
    metrics = model.fit(df_train, freq="H", validation_df=df_test)

    return model, metrics


# def train_model(df1):
#     df1s = df1[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
#     df1s.loc[:, "rain"] = df1s["rain"].astype(int)
#     df1s.loc[:, "snow"] = df1s["snow"].astype(int)

#     model = NeuralProphet(
#         n_forecasts=1,
#         n_lags=24,
#         num_hidden_layers=1,
#         d_hidden=16,
#         learning_rate=0.01,
#         drop_missing=True,
#         n_changepoints=100
#     )

#     model.add_lagged_regressor(names = list(df1s)[2:])
#     # model.add_future_regressor(name = "temp")
#     # model.add_future_regressor(name = "rh")
#     # model.add_future_regressor(name = "pressure")
#     # model.add_future_regressor(name = "windspeed")
#     # model.add_future_regressor(name = "rain")
#     # model.add_future_regressor(name = "snow")
#     df_train, df_test = model.split_df(df1s, freq="H", valid_p=0.10)
#     metrics = model.fit(df_train, freq="H", validation_df=df_test)

#     return model
#     # ## save results
#     # save(model, 'checkpoints/np_lag24_AR_weights_f24_'+df1.zone.values[0]+'.pth')
#     # metrics['zone'] = df1.zone.values[0]
#     # metrics.to_csv("checkpoints/metrics_f24_"+df1.zone.values[0]+'.csv')
#     # return metrics


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
    res = DF.groupby("zone").apply(run_model)
    
    # ## parallel by pyspark
    # res = df.groupby("zone").applyInPandas(
    #     train_model,
    #     schema="SmoothL1Loss double, MAE double, RMSE double, Loss double, RegLoss double, SmoothL1Loss_val double, MAE_val double, RMSE_val double, zone string"
    # ).toPandas()

    res.to_csv("checkpoints_f24/np_mape_results.csv")

    # ## eval model
    # test_res=[]
    # for z in DF.zone.unique():
    #     df1 = DF.loc[DF.zone==z]
    #     df1s = df1[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
    #     df1s.loc[:, "rain"] = df1s["rain"].astype(int)
    #     df1s.loc[:, "snow"] = df1s["snow"].astype(int)
    #     model = load("checkpoints/np_lag24_AR_weights_f1_"+z+".pth")
    #     pred1 = model.predict(df1s)
    #     pred1 = pd.merge(df1, pred1, on="ds")
    #     test1 = pred1.loc[pred1.test==1]
    #     res1 = mape(test1.y_x, test1.yhat1)
    #     test_res.append(res1)

    # res_df = pd.DataFrame({"zone":DF.zone.unique(), "mape": test_res})
    # res_df.to_csv("checkpoints/test_mape_zones_f1.csv")
