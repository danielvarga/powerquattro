# based on https://unit8co.github.io/darts/examples/01-darts-intro.html

import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import reduce

from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT
)

from darts.metrics import mape, mase
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.datasets import AirPassengersDataset

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)






data = pd.read_csv("terheles_fixed.tsv", sep="\t")

data['Date'] = pd.to_datetime(data['Korrigált időpont'])
data['Consumption'] = data['Hatásos teljesítmény']
data = data[['Date', 'Consumption']]

from pandas.tseries.offsets import DateOffset
'''
data['Date2'] = data['Date'] + DateOffset(minutes=15)
lastrow = None
for indx, row in data.iterrows():
    if lastrow is not None:
        print(row['Date'], lastrow['Date2'])
    lastrow = row
exit()
'''

data.plot(x='Date', y='Consumption')
plt.show()


data = data.set_index('Date')
data = data.sort_index() # https://bobhowto.wordpress.com/2020/04/11/why-i-hate-python-valueerror-index-must-be-monotonic-increasing-or-decreasing/
print(data.head())
print("====")
data = data.asfreq(freq=DateOffset(minutes=15), method='backfill')
print(data.head())


series = TimeSeries.from_series(data['Consumption'])

print("====")
print(series[:10])

# series = AirPassengersDataset().load()


# 4
train, val = series.split_before(pd.Timestamp('20200101'))
print("restricting datasets, darts is slooow")
train = train[:4*24*14]
val = val[:4*24*14]

train.plot(label='training')
val.plot(label='validation')
plt.legend()
plt.show()


# 6
plot_acf(train[:4*24*50], max_lag = 4*24*14, alpha = .05)
plt.show()


# 11
def eval_model(model):
    model.fit(train)
    forecast = model.predict(len(val))
    print('model {} obtains MAPE: {:.2f}%'.format(model, mape(val, forecast)))


# does not work, cries about 'ValueError: Invalid frequency: <DateOffset: minutes=15>'
eval_model(ExponentialSmoothing(seasonal_periods=4*24*7))
eval_model(Prophet())
eval_model(AutoARIMA())
eval_model(Theta())


# 20
# commented out because inference with many samples is super slow, no vectorization whatsoever.
'''
model_es = ExponentialSmoothing(seasonal_periods=4*24*7)
model_es.fit(train)
probabilistic_forecast = model_es.predict(len(val), num_samples=5)

series.plot(label='actual')
probabilistic_forecast.plot(low_quantile=0.01, high_quantile=0.99, label='1-99th percentiles')
probabilistic_forecast.plot(low_quantile=0.2, high_quantile=0.8, label='20-80th percentiles')
# probabilistic_forecast.plot(label='probabilistic forecast')
plt.legend()
plt.show()
'''

# 26
# not working
ensemble_model = RegressionEnsembleModel(
    forecasting_models=[NaiveSeasonal(4*24), NaiveSeasonal(4*24*7), NaiveDrift()],
    regression_train_n_points=12)

ensemble_model.fit(train)
ensemble_pred = ensemble_model.predict(36)

series.plot(label='actual')
ensemble_pred.plot(label='Ensemble forecast')
plt.title('MAPE = {:.2f}%'.format(mape(ensemble_pred, series)))
plt.legend()
plt.show()


# 17 kinda
print('doing residuals analysis')
plot_residuals_analysis(model_es.residuals(series))
plt.show()
