from prophet import Prophet
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse
from pyspark.ml.feature import Imputer
import pyarrow
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
from prophet.serialize import model_to_json, model_from_json
from pyspark.sql.functions import pandas_udf, PandasUDFType


spark = SparkSession.builder.getOrCreate()


def load_data():
    df = spark.read.csv("Data/03_full_dataset_clean.csv", header=True, inferSchema=True)
    df = df.select("zone", "dt", "mw", "temp", "precip", "rh", "pressure",
                   "windspeed", "rain", "snow", "test")
    df = df.withColumnRenamed("dt", "ds").withColumnRenamed("mw", "y")

    # imputation
    cols = ["precip", "rh", "pressure", "windspeed"]
    imputer = Imputer(strategy="mean", inputCols=cols, outputCols=cols)
    df = imputer.fit(df).transform(df)

    df.printSchema()
    ## df.groupby("zone").count().show()
    return df
    

def run_model(df1):
    ## tune model
    cutoffs = pd.to_datetime(['2022-01-15', '2022-03-15', '2022-05-15', '2022-8-15'])

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.2],  #[0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [
        dict(zip(param_grid.keys(), v))
        for v in itertools.product(*param_grid.values())
    ]
    mape = []
    for params in all_params:
        model = Prophet(**params) #.add_seasonality("daily", period=1, fourier_order=1, mode="multiplicative")

        model.add_regressor("temp")
        model.add_regressor("precip")
        model.add_regressor("rh")
        model.add_regressor("pressure")
        model.add_regressor("windspeed")
        model.add_regressor("rain")
        model.add_regressor("snow")

        model.fit(df1)
        df_cv = cross_validation(model,
                                 cutoffs=cutoffs,
                                 horizon='7 days',
                                 parallel="processes")
        
        df_p = performance_metrics(df_cv, metrics=["mape"], rolling_window=1)
        mape.append(df_p['mape'].values[0])

    # Find the best parameters
    ## print(mape)
    tuning_results = pd.DataFrame(all_params)
    tuning_results['mape'] = mape
    tuning_results['zone'] = df1.zone[0]
    tuning_results.to_csv("checkpoints/"+df1.zone[0]+"_CV.csv")
    
    idx = tuning_results.mape.argmin()
    params = {'changepoint_prior_scale': tuning_results.iloc[idx, 0],
              'seasonality_prior_scale': tuning_results.iloc[idx, 1],
              'seasonality_mode': tuning_results.iloc[idx, 2]}

    metrics = train_model(df1, params)

    return metrics


def train_model(df1, params):
    model = Prophet(**params)
    model.add_regressor("temp")
    model.add_regressor("precip")
    model.add_regressor("rh")
    model.add_regressor("pressure")
    model.add_regressor("windspeed")
    model.add_regressor("rain")
    model.add_regressor("snow")

    model.fit(df1)

    with open('checkpoints/prophet_model_'+df1.zone[0]+'.json', 'w') as fout:
        fout.write(model_to_json(model))

    test = df1.tail(72)
    forecast_tmp = model.predict(test)
    err = mape(test.y, forecast_tmp.yhat)

    err2 = eval_model(df1, model).mape[0]
    return pd.DataFrame({'zone': [df1.zone[0]], 'mape_test': [err2], 'mape_f': [err]})


def eval_model(df1, model):
    # with open('checkpoints/prophet_model_'+df1.zone[0]+'.json', 'r') as fin:
    #     m = model_from_json(fin.read())

    test = df1.loc[df1.test==1]
    pred = model.predict(test)
    err1 = mape(test.y, pred.yhat)
    print("test MAPE: {}".format(err1))
    
    return pd.DataFrame({'zone': [df1.zone[0]], 'mape': [err1]})


if __name__ == "__main__":
    df = load_data()
    zones = df.select('zone').distinct().collect()
    
    # ## tune hyperparameters using CV by AE zone
    # df1 = df.filter((col("zone") == "AEP"))
    # df1 = df1.toPandas()

    # res1 = tune_model(df1)
    # res1.to_csv("checkpoints/prophet_CV.csv")
    
    # idx = res1.mape.argmin()
    # params = {'changepoint_prior_scale': res1.iloc[idx, 0], 'seasonality_prior_scale': res1.iloc[idx, 1], 'seasonality_mode': res1.iloc[idx, 2]}

    # err = train_model(df1, params)

    # ## train models
    # def train_model_s(df1):
    #     return train_model(df1, params)

    # df.groupby("zone").applyInPandas(
    #     train_model_s,
    #     schema="zone string, mape double"
    # ).show(21)

    # ## eval
    # mapes = df.groupby("zone").applyInPandas(
    #     eval_model,
    #     schema="zone string, mape double"
    # ).toPandas()
    # mapes.to_csv("checkpoints/prophet_eval.csv")

    ## run model
    res = df.groupby("zone").applyInPandas(
        run_model,
        schema="zone string, mape_test double, mape_f double"
    ).toPandas()

    ## tune > param => train => eval
