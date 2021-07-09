import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


# AUS dataset, half-hours resolution, single value, 300 households.
# data = pd.read_csv("2012-2013-Solar-home-electricity-data-v2.csv", sep=",")

# GERMAN dataset, see https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1
# data = pd.read_csv("opsd_germany_daily.csv", sep=",")

# Hungarian dataset
data = pd.read_csv("terheles_fixed.tsv", sep="\t")

data['Date'] = pd.to_datetime(data['Korrigált időpont'])
data = data.set_index('Date')
c = data['Consumption'] = data['Hatásos teljesítmény']


'''
seaborn.violinplot(c)
plt.show()

n = len(c)

plt.plot(c[:1000])
plt.show()


fft = np.fft.rfft(c)
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(c)
fifteens_per_year = 24*4*365.2524
years_per_dataset = n_samples_h/(fifteens_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 52.1789, 365.2524], labels=['1/Year', '1/week', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
plt.show()
'''


# creating new dataframe from consumption column
dc = data[['Consumption']]
# inserting new column with yesterday's consumption values
for i in range(1, 4 * 24 + 1):
    dc.loc[:, 'd%02d' % i] = dc.loc[:,'Consumption'].shift(4 * 24 + i) # t-2days to t-1day
    dc.loc[:, 'w%02d' % i] = dc.loc[:,'Consumption'].shift(7 * 4 * 24 + i) # t-7days to t-8days
# dc.loc[:,'prev'] = dc.loc[:,'Consumption'].shift(4 * 24)
# dc.loc[:,'prev2'] = dc.loc[:,'Consumption'].shift(7 * 4 * 24)
# inserting another column with difference between yesterday and day before yesterday's consumption values.
# dropping NAs
dc = dc.dropna()

X_train = dc[:'2019'].drop(['Consumption'], axis = 1)
y_train = dc.loc[:'2019', 'Consumption']
X_test = dc['2020'].drop(['Consumption'], axis = 1)
y_test = dc.loc['2020', 'Consumption']

models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver = 'lbfgs', max_iter=20)))  #neural network
models.append(('KNN', KNeighborsRegressor())) 
models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees


# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # TimeSeries Cross validation
    tscv = TimeSeriesSplit(n_splits=2)

    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    if name == 'NN':
        model.fit(X_train, y_train)
        y_true = y_train.values
        y_pred = model.predict(X_train)
        plt.plot(y_true[1000:2000])
        plt.plot(y_pred[1000:2000])
        plt.title("In sample forecast")
        plt.xlabel("Date")
        plt.ylabel("Consumption")
        plt.show()


        y_true = y_test.values
        y_pred = model.predict(X_test)
        plt.plot(y_true[1000:2000])
        plt.plot(y_pred[1000:2000])
        plt.title("Out of sample forecast")
        plt.xlabel("Date")
        plt.ylabel("Consumption")
        plt.show()

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
