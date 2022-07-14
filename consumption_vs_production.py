import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


prod = pd.read_csv('fot_full.csv', sep=';', skipinitialspace=True, na_values='n/a')

prod['Time'] = (prod['DATE'] + ' ' + prod['TIME']).str.strip()
date_time = prod['Time'] = pd.to_datetime(prod['Time'], format='%Y. %m. %d %H:%M:%S') # 2021. 04. 20;02:02:00

# some time zone sh*t, this is needed for good correspondence between the two:
# date_time = date_time - pd.DateOffset(hours=2)


prod = prod[['Time', 'TREND_AKKUBANK_PV1_P', 'TREND_AKKUBANK_PV2_P']]

prod = prod.rename({'TREND_AKKUBANK_PV1_P': 'pv1', 'TREND_AKKUBANK_PV2_P': 'pv2'}, axis='columns')


daily_aggregate = prod.groupby(pd.Grouper(key='Time', freq='D')).sum()
plt.hist(daily_aggregate['pv1'], bins=100)
plt.xlabel('Power output in some unknown dimension')
plt.ylabel('Number of days with output in that range')
plt.show()


month_to_season = np.array([
    None,
    'Winter', 'Winter',
    'Spring', 'Spring', 'Spring',
    'Summer', 'Summer', 'Summer',
    'Autumn', 'Autumn', 'Autumn',
    'Winter'
])

season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

daily_aggregate['season'] = month_to_season[daily_aggregate.index.month]


seasons_aggregate = daily_aggregate.groupby('season').sum()
print(seasons_aggregate)


grouped = []
labels = []


for season in season_order:
    grouped.append(daily_aggregate.groupby('season').get_group(season)['pv1'].to_numpy())

plt.hist(grouped, histtype='bar', stacked=True, bins=50, label=season_order)
plt.legend()
plt.xlabel('Power output in some unknown dimension')
plt.ylabel('Number of days with output in that range, colored by season')
plt.show()


for season in season_order:
    pv1 = daily_aggregate.groupby('season').get_group(season)['pv1'].to_numpy()
    plt.hist(pv1, bins=50)
    plt.xlim((0, daily_aggregate['pv1'].max()))
    plt.title(season)
    plt.show()


exit()


# date_time -= pd.to_timedelta('365 days')
# date_time -= pd.to_timedelta('365 days')


# print(prod[prod['Time'] < '2021-01-01'])

# prod_filtered = prod[(prod['Time'] > '2020-08-19') & (prod['Time'] < '2020-08-21')]

# prod_filtered = prod[(prod['Time'] > '2020-08-21') & (prod['Time'] < '2020-08-23')]
prod_filtered = prod


plt.plot(prod_filtered['Time'], prod_filtered['pv1'])

d = prod_filtered['pv1'].to_numpy()
d = np.nan_to_num(d)
d_smoothed = np.zeros_like(d)
window = 10
for i in range(len(d) - window):
    d_smoothed[i + window // 2] = d[i: i+window].max()

plt.plot(prod_filtered['Time'], d_smoothed)

print("total original", d.sum())
print("total smoothed", d_smoothed.sum())
print("ratio", d.sum() / d_smoothed.sum())

plt.show()
exit()


'''
prod = pd.read_csv("fot_meteo.csv", sep='\t')

prod = prod[['Time', 'TREND_AKKUBANK_PV1_P', 'TREND_AKKUBANK_PV2_P', 'ta', 'sg', 'sr', 'suv']]

prod = prod.rename({'TREND_AKKUBANK_PV1_P': 'pv1', 'TREND_AKKUBANK_PV2_P': 'pv2'}, axis='columns')

date_time = pd.to_datetime(prod['Time'], format='%Y-%m-%d %H:%M:%S') # 2021-01-01 00:00:00

date_time -= pd.to_timedelta('365 days')
date_time -= pd.to_timedelta('365 days')
'''


n = 5000

# TODO 200/8!
plt.plot(date_time[:n], prod['pv1'][:n] * 200 / 8)

data = pd.read_csv("terheles_fixed.tsv", sep="\t")

data['Date'] = pd.to_datetime(data['Korrigált időpont'])
# data = data.set_index('Date')
c = data['Consumption'] = data['Hatásos teljesítmény']

plt.plot(data['Date'][:n], c[:n])
plt.show()
