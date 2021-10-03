import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# met columns:
# sg: 'gamma dózis'
# sr: 'globálsugárzás'
# suv: 'UV-sugárzás'
# ta: 'hőmérséklet'


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

plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 20000
ax.scatter(df['ta'][:n], df['sr'][:n], df['pv1'][:n], s=20, alpha=0.1)
plt.show()
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


X_train = X[::2]
Y_train = Y[::2]
X_test = X[1::2]
Y_test = Y[1::2]


model = Sequential()
input_shape = (len(indeps), )
output_dim = 2 # 'pv1', 'pv2'
hidden_dim = 50
model.add(Dense(hidden_dim, input_shape=input_shape, activation='relu'))
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dense(output_dim, activation='linear'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01))


model.fit(X_train, Y_train, epochs=200, batch_size=50, verbose=1, validation_split=0.2)

Y_predict = model.predict(X_test)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot(x, y):
    target_n = 10000
    if len(x) > target_n:
        keep = np.random.choice(len(x), size=target_n, replace=False)
        x = x[keep]
        y = y[keep]
    # ta, sr, pv1
    ax.scatter(x[:, 0], x[:, 2], y[:, 0], s=20, alpha=0.1)

plot(X_test, Y_test)
plot(X_test, Y_predict)
plt.show()
