import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# python solar_characteristic.py fot_meteo.csv


# met columns:
# sg: 'gamma dózis'
# sr: 'globálsugárzás'
# suv: 'UV-sugárzás'
# ta: 'hőmérséklet'


prefix = "solar_characteristic"

suffix = 1


def save_and_show():
    global suffix
    plt.savefig(f"{prefix}-{suffix:02}.png", dpi=600)
    # plt.show()
    plt.clf()
    suffix += 1


csv_path, = sys.argv[1:]

df = pd.read_csv(csv_path, sep='\t')
print(df.head())

df = df[['Time', 'TREND_AKKUBANK_PV1_P', 'TREND_AKKUBANK_PV2_P', 'ta', 'sg', 'sr', 'suv']]

df = df.rename({'TREND_AKKUBANK_PV1_P': 'pv1', 'TREND_AKKUBANK_PV2_P': 'pv2'}, axis='columns')

print(df.head())

date_time = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') # 2021-01-01 00:00:00

timestamp_s = date_time.map(pd.Timestamp.timestamp)


# Similar to the wind direction, the time in seconds is not a useful model input. Being weather data, it has clear daily and yearly periodicity. There are many ways you could deal with periodicity.
# 
# You can get usable signals by using sine and cosine transforms to clear "Time of day" and "Time of year" signals:

# In[13]:


day = 24*60*60
year = (365.2425)*day

df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))



'''
plot_cols = ['ta', 'sr', 'pv1', 'pv2']
plot_features = df[plot_cols]
plot_features.index = date_time

print("======")
print(plot_features['pv2'].head())

_ = plot_features.plot(subplots=True)

save_and_show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 20000
ax.scatter(df['ta'][:n], df['sr'][:n], df['pv1'][:n], s=20, alpha=0.1)
save_and_show()
'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras


met_indeps = ['ta', 'sg', 'sr', 'suv']
time_indeps = ['day_sin', 'day_cos', 'year_sin', 'year_cos']
indeps = met_indeps + time_indeps
X = df[indeps].to_numpy()
Y = df[['pv1', 'pv2']].to_numpy()
X = X[:20000]
Y = Y[:20000]

keep = ~ np.isnan(Y).any(axis=1)
X = X[keep, :]
Y = Y[keep, :]

# random shuffle!!!
perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]
n = len(X)

rmse = ((Y[n // 2:, 1] - Y[n // 2:, 0]) ** 2).mean() ** 0.5
mae = (np.abs(Y[n // 2:, 1] - Y[n // 2:, 0])).mean()
print("'pv1=pv2' baseline on test: rmse", rmse, "mae", mae)


# only pv1 from now on!!!
Y = Y[:, 0]

X_train = X[:n // 2]
Y_train = Y[:n // 2]
X_test = X[n // 2:]
Y_test = Y[n // 2:]


rmse = ((Y_test[1:] - Y_test[:-1]) ** 2).mean() ** 0.5
mae = (np.abs(Y_test[1:] - Y_test[:-1])).mean()
print("'repeat last' baseline on test: rmse", rmse, "mae", mae)

import sklearn.ensemble
reg = sklearn.ensemble.GradientBoostingRegressor(loss='lad')
reg.fit(X_train, Y_train)
Y_predict = reg.predict(X_test)
rmse = ((Y_test - Y_predict) ** 2).mean() ** 0.5
mae = (np.abs(Y_test - Y_predict)).mean()
print("GBR on test: rmse", rmse, "mae", mae)


model = Sequential()
input_shape = (len(indeps), )
output_dim = 1 # 'pv1', 'pv2'
hidden_dim = 100 # 50
model.add(Dense(hidden_dim, input_shape=input_shape, activation='relu'))
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dense(output_dim, activation='linear'))
model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[keras.losses.MeanSquaredError()])


model.fit(X_train, Y_train, epochs=200, batch_size=50, verbose=1, validation_split=0.2)

Y_predict = model.predict(X_test)[:, 0]

rmse = ((Y_test - Y_predict) ** 2).mean() ** 0.5
mae = (np.abs(Y_test - Y_predict)).mean()
print("MLP on test: rmse", rmse, "mae", mae)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot(x, y):
    target_n = 2000
    if len(x) > target_n:
        keep = np.random.choice(len(x), size=target_n, replace=False)
        x = x[keep]
        y = y[keep]
    # ta, sr, pv1
    ax.scatter(x[:, 0], x[:, 2], y, s=20, alpha=0.1)

plot(X_test, Y_test)
plot(X_test, Y_predict)
save_and_show()


for i in range(len(indeps)):
    plt.scatter(X_test[:, i], Y_test)
    plt.scatter(X_test[:, i], Y_predict)
    plt.title(indeps[i] + ' vs pv1')
    save_and_show()
