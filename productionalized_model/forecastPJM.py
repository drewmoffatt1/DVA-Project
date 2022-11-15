import pandas as pd
import numpy as np
from sklearn import linear_model

def prepData(dataFile):
    df = pd.read_csv(dataFile) #'03_full_dataset_clean.csv'
    return df
    #X = df[['temp', 'precip', 'pressure','windspeed','rain','snow']]
    
def buildModels(df):
    modelsList = []
    for z in df.zone.unique():
        ## grab training data
        zoneData = df.loc[df.zone==z]
        zoneData = zoneData.dropna()
        zoneData["previous_mw"] = zoneData["mw"].shift(48)
        zoneData["previous_temp"] = zoneData["temp"].shift(24)
        zoneData = zoneData.dropna()
        trainData = zoneData.loc[zoneData.test==0]
        testData = zoneData.loc[zoneData.test==1]
        
        ## create model with training data
        x = trainData[['month','temp','precip','rh','pressure','windspeed','rain','snow']]
        y = trainData['mw']
    
        regr = linear_model.LinearRegression()
        modelsList.append(regr.fit(x, y))
    # gather models
    model_df = pd.DataFrame({"zone":df.zone.unique(), "model": modelsList})
    return model_df

def getNewData():
    manualDPData = [[10,55.0,0.0,83.0,29.68,0.0,False,False]]
    manualDF = pd.DataFrame(manualDPData, columns=['month','temp','precip','rh','pressure','windspeed','rain','snow'])
    return manualDF

def predMwUsage(dataPoint,models):
    #manualDPData = [[10,55.0,0.0,83.0,29.68,0.0,False,False]]
    #manualDF = pd.DataFrame(manualDPData, columns=['month','temp','precip','rh','pressure','windspeed','rain','snow'])
    predValue = models.iloc[0].model.predict(dataPoint)
    return predValue[0]

def runStuff():
    df = prepData('03_full_dataset_clean.csv')
    model_df = buildModels(df)
    newData = getNewData()
    value = predMwUsage(newData,model_df)
    #predMwUsage(newData,model_df)
    print(value)
    return value

runStuff()