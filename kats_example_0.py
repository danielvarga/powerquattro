
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

import warnings
warnings.filterwarnings('ignore')

from kats.consts import TimeSeriesData

# import the param and model classes for Prophet model
from kats.models.prophet import ProphetModel, ProphetParams


air_passengers_df = pd.read_csv("air_passengers.csv")
air_passengers_df.columns = ["time", "value"]
print(air_passengers_df.head())
air_passengers_ts = TimeSeriesData(air_passengers_df)
print("====")
print(air_passengers_ts[:10])


# create a model param instance
params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results

# create a prophet model instance
m = ProphetModel(air_passengers_ts, params)

# fit model simply by calling m.fit()
m.fit()

# make prediction for next 30 month
fcst = m.predict(steps=30, freq="MS")

print(fcst.head())

m.plot()
plt.show()
