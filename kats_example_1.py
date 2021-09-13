
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import reduce

from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams


data = pd.read_csv("terheles_fixed.tsv", sep="\t")

data['Date'] = pd.to_datetime(data['Korrigált időpont'])
data['Consumption'] = data['Hatásos teljesítmény']
data = data[['Date', 'Consumption']]

data.columns = ["time", "value"]
print(data.head())


'''
data.plot(x='Date', y='Consumption')
plt.show()
'''

series = TimeSeriesData(data, time_col_name='time')

print("====")
print(series[:10])


split_position = len(series) // 2 # was pd.Timestamp('20200101')
train, val = series[:split_position], series[split_position:]

print("restricting datasets, darts is slooow")
train = train[:4*24*7*5]
val = val[:4*24*7*5]

train.plot(cols=['value'])
val.plot(cols=['value'])
plt.show()


params = ProphetParams(seasonality_mode='multiplicative')
m = ProphetModel(train, params)
m.fit()

forecast = m.predict(steps=4*24*7*5, freq="15min")
print(forecast.head())
forecast['fcst'].plot()
plt.show()
