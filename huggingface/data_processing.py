from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d   


PATH_PREFIX = "./"

START = f"2021-01-01"
END = f"2022-01-01"


def read_datasets(mini=False):
    if mini:
        met_filename = 'PL_44527.2101.csv.gz'
        cons_filename = 'pq_terheles_202101_adatok.tsv'
    else:
        met_filename = 'PL_44527.19-21.csv.gz'
        cons_filename = 'pq_terheles_2021_adatok.tsv'

    #@title ### Preprocessing meteorologic data
    met_data = pd.read_csv(PATH_PREFIX + met_filename, compression='gzip', sep=';', skipinitialspace=True, na_values='n/a', skiprows=[0, 1, 2, 3, 4])
    met_data['Time'] = met_data['Time'].astype(str)
    date_time = met_data['Time'] = pd.to_datetime(met_data['Time'], format='%Y%m%d%H%M')
    met_data = met_data.set_index('Time')


    #@title ### Preprocessing consumption data
    cons_data = pd.read_csv(PATH_PREFIX + cons_filename, sep='\t', skipinitialspace=True, na_values='n/a', decimal=',')
    cons_data['Time'] = pd.to_datetime(cons_data['Korrigált időpont'], format='%m/%d/%y %H:%M')
    cons_data = cons_data.set_index('Time')
    cons_data['Consumption'] = cons_data['Hatásos teljesítmény [kW]']

    # consumption data is at 14 29 44 59 minutes, we move it by 1 minute
    # to sync it with production data:
    cons_data.index = cons_data.index + pd.DateOffset(minutes=1)

    met_2021_data = met_data[(met_data.index >= START) & (met_data.index < END)]
    cons_2021_data = cons_data[(cons_data.index >= START) & (cons_data.index < END)]

    return met_2021_data, cons_2021_data


@dataclass
class Parameters:
    solar_cell_num: float = 114 # units
    solar_efficiency: float = 0.93 * 0.96 # [dimensionless]
    NOCT: float = 280 # [W]
    NOCT_irradiation: float = 800 # [W/m^2]

    bess_nominal_capacity: float = 330 # [Ah]
    bess_charge: float = 50 # [kW]
    bess_discharge: float = 60 # [kW]
    voltage: float = 600 # [V]
    maximal_depth_of_discharge: float = 0.75 # [dimensionless]
    energy_loss: float = 0.1 # [dimensionless]
    bess_present: bool = True # [boolean]

    @property
    def bess_capacity(self):
        return self.bess_nominal_capacity * self.voltage / 1000


# mutates met_2021_data
def add_production_field(met_2021_data, parameters):
    sr = met_2021_data['sr']

    nop_total = sr * parameters.solar_cell_num * parameters.solar_efficiency * parameters.NOCT / parameters.NOCT_irradiation / 1e3
    nop_total = nop_total.clip(0)
    met_2021_data['Production'] = nop_total


def interpolate_and_join(met_2021_data, cons_2021_data):
    applicable = 24*60*365 - 15 + 5

    demand_f = interp1d(range(0, 365*24*60, 15), cons_2021_data['Consumption'])
    #demand_f = interp1d(range(0, 6*24*60, 15), cons_2021_data['Consumption'])
    demand_interp = demand_f(range(0, applicable, 5))

    production_f = interp1d(range(0, 365*24*60, 10), met_2021_data['Production'])
    #production_f = interp1d(range(0, 6*24*60, 10), met_2021_data['Production'])
    production_interp = production_f(range(0, applicable, 5))

    all_2021_datetimeindex = pd.date_range(start=START, end=END, freq='5min')[:len(production_interp)]

    all_2021_data = pd.DataFrame({'Consumption': demand_interp, 'Production': production_interp})
    all_2021_data = all_2021_data.set_index(all_2021_datetimeindex)
    return all_2021_data


# TODO build a dataframe instead
def monthly_analysis(results):
    consumptions = []
    for month in range(1, 13):
        start = f"2021-{month:02}-01"
        end = f"2021-{month+1:02}-01"
        if month == 12:
            end = "2022-01-01"
        results_in_month = results[(results.index >= start) & (results.index < end)]

        total = results_in_month['Consumption'].sum()
        network = results_in_month['consumption_from_network'].sum()
        solar = results_in_month['consumption_from_solar'].sum()
        bess = results_in_month['consumption_from_bess'].sum()
        consumptions.append([network, solar, bess])

    consumptions = np.array(consumptions)
    step_in_minutes = results.index.freq.n
    # consumption is given in kW. each tick is step_in_minutes long (5mins, in fact)
    # we get consumption in kWh if we multiply sum by step_in_minutes/60
    consumptions_in_mwh = consumptions * (step_in_minutes / 60) / 1000
    return consumptions_in_mwh
