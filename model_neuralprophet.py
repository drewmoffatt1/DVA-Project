from neuralprophet import NeuralProphet
from model_prophet import load_data##, eval_model
import pandas as pd
import numpy as np
from datetime import datetime
import pyspark
import torch
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# df = pd.read_csv("Data/03_full_dataset_clean.csv")
# df = df.rename(columns = {"dt":"ds", "mw":"y"})
# df.ds = pd.to_datetime(df.ds, utc=True)
# df.ds = df.ds.apply(lambda x: str(x)[:-6])
# df.ds = pd.to_datetime(df.ds)

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("Data/03_full_dataset_clean.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("dt", "ds").withColumnRenamed("mw", "y")


df1 = df.filter(col("zone") == "AE").toPandas()
df1.loc[df1.ds.isin(df1.ds[df1.ds.duplicated()])]

df1 = df1.drop_duplicates(subset=['ds'])
df1s = df1[["ds", "y", "temp", "rh", "pressure", "windspeed", "rain", "snow"]]
df1s["rain"] = df1s["rain"].astype(int)
df1s["snow"] = df1s["snow"].astype(int)

m = NeuralProphet(
        yearly_seasonality=3,
        weekly_seasonality=1,
        daily_seasonality=8,
        learning_rate=0.1,
    )

m.add_future_regressor(name="temp")
m.add_future_regressor(name="rh")
m.add_future_regressor(name="pressure")
m.add_future_regressor(name="windspeed")
m.add_future_regressor(name="rain")
m.add_future_regressor(name="snow")

df_train, df_test = m.split_df(df1s, freq="H", valid_p=0.10)

metrics = m.fit(df_train, freq="H", validation_df=df_test)

torch.save(m.model.state_dict(), 'checkpoints/np_t1_weights.pth')

test = df1s.loc[df1.test==1]
pred = m.predict(test)
err1 = mape(pred.y, pred.yhat1)

m.predict()

if __name__ == "__main__":

    load_data()
