#!/usr/bin/env python
# coding: utf-8

# cat fot_adatok/PQ\ akku_window1_part10.csv | tr ',' '.' | sed "s/ *;/;/g" | sed "s/; */;/g" > fot_proba.csv
# ( cat fot_adatok/PQ\ akku_window1_part102.csv | head -1 ; for ((i=101; i>=0; --i)) ; do cat fot_adatok/PQ\ akku_window1_part$i.csv | tail -n +2 ; done ) | tr ',' '.' | sed "s/ *;/;/g" | sed "s/; */;/g" > fot_full.csv
# -> this is a bit unsorted on two dates 2019-10-25 and 2019-10-27, that's in fot_adatok/PQ akku_window1_part88.csv . only there.
# python join.py pestszentlorinc.csv fot_full.csv fot_meteo.csv


import sys
import os
import datetime

import numpy as np
import pandas as pd


met_filename, pow_filename, joined_filename = sys.argv[1:]

met_data = pd.read_csv(met_filename, sep=';', skipinitialspace=True)

pow_data = pd.read_csv(pow_filename, sep=';', skipinitialspace=True)

met_data['Time'] = pd.to_datetime(met_data['Time'], format='%Y%m%d%H%M')
met_data = met_data.set_index('Time')

# pestszentlorinc.csv has 10 minute steps, 2021-01-01 00:00 to 2021-07-04 23:50
print(met_data.head())
print("====")

pow_data['Time'] = (pow_data['DATE'] + ' ' + pow_data['TIME']).str.strip()
date_time = pow_data['Time'] = pd.to_datetime(pow_data['Time'], format='%Y. %m. %d %H:%M:%S') # 2021. 04. 20;02:02:00

# some time zone sh*t, this is needed for good correspondence between the two:
date_time = date_time - pd.DateOffset(hours=2)

pow_data = pow_data.set_index(date_time)

# fot_proba.csv has 1 minute steps, 2021-04-18 01:17 to 2021-04-24 23:56
print(pow_data.head())

# met columns:
# sg: 'gamma dózis'
# sr: 'globálsugárzás'
# suv: 'UV-sugárzás'
# ta: 'hőmérséklet'


start = max((met_data.index[0], pow_data.index[0]))
finish = min((met_data.index[-1], pow_data.index[-1]))
print("overlap:", start, finish)
pow_data = pow_data.loc[start:finish]
met_data = met_data.loc[start:finish]
# we need the larger granularity of the two.
start = met_data.index[0]
finish = met_data.index[-1]
pow_data = pow_data.loc[start:finish]

# we're throwing away all that valuable data, ouch.
pow_data = pow_data.loc[::10]

sp = set(pow_data.index)
sm = set(met_data.index)
print("overlap", len(sp), len(sm), len(sp.intersection(sm)))
assert len(sp.intersection(sm)) > 0.99 * max((len(sp), len(sm)))

df = pow_data.join(met_data, how='outer')
print(df.head())

df.to_csv(open(joined_filename, 'w'), sep='\t')

import matplotlib.pyplot as plt

df['sr'].plot()
(60 * df['TREND_AKKUBANK_PV1_P']).plot()
plt.show()

