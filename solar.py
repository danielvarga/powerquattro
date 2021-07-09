import pandas as pd
import matplotlib.pyplot as plt
import sys

# cat fot_adatok/PQ\ akku_window1_part10.csv | tr ',' '.' | sed "s/ *;/;/g" | sed "s/; */;/g" > fot_proba.csv
# python habp.py pestszentlorinc.csv fot_proba.csv

met_filename, pow_filename = sys.argv[1:]

met_data = pd.read_csv(met_filename, sep=';', skipinitialspace=True)

pow_data = pd.read_csv(pow_filename, sep=';', skipinitialspace=True)

met_data['Time'] = pd.to_datetime(met_data['Time'], format='%Y%m%d%H%M')
met_data = met_data.set_index('Time')

pow_data['Time'] = (pow_data['DATE'] + ' ' + pow_data['TIME']).str.strip()
pow_data['Time'] = pd.to_datetime(pow_data['Time'], format='%Y. %m. %d %H:%M:%S') # 2021. 04. 20;02:02:00
pow_data['Time'] = pow_data['Time'] - pd.DateOffset(hours=2)
pow_data = pow_data.set_index('Time')


'''
met_data['ta'].plot()
plt.title("10 minutes average temperature at Pestszentlőrinc")
plt.show()
'''

# met_data['sg'].plot(label='gamma dózis')
met_data['sr'].plot(label='globálsugárzás')
# met_data['suv'].plot(label='UV-sugárzás')

# print(pow_data['TREND_AKKUBANK_AKKU_P'])
# pow_data['TREND_AKKUBANK_AKKU_P'] = pow_data['TREND_AKKUBANK_AKKU_P'].astype(float)
# print(pow_data['TREND_AKKUBANK_AKKU_P'])

(pow_data['TREND_AKKUBANK_PV1_P'] * 30).plot(label='Napelem 1 teljesítmény')
(pow_data['TREND_AKKUBANK_PV2_P'] * 30).plot(label='Napelem 2 teljesítmény')

plt.title("Sugárzási és naperőmű adatok")
plt.legend()
plt.show()


plt.scatter(pow_data['TREND_AKKUBANK_PV1_P'], pow_data['TREND_AKKUBANK_PV2_P'], c=range(len(pow_data)))
plt.show()


plt.scatter(met_data['sr'], met_data['suv'])
plt.show()
