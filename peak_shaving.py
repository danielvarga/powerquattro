import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

prefix = "peak_shaving"

suffix = 1


def save_and_show():
    global suffix
    plt.savefig(f"{prefix}-{suffix:02}.png", dpi=600)
    # plt.show()
    plt.clf()
    suffix += 1


data = pd.read_csv("terheles_fixed.tsv", sep="\t")


'''
# verify that Ad hatasos [kwh] is simply hatasos teljesitmeny / 4.
plt.scatter(data['Ad Hatásos [kWh]'][:1000], data['Hatásos teljesítmény'][:1000])
plt.show()
'''

data['Date'] = pd.to_datetime(data['Korrigált időpont'])
data = data.set_index('Date')
c = data['Consumption'] = data['Ad Hatásos [kWh]']


n = len(c)

plt.title("Consumption, kWh")
plt.plot(c[:1000])
save_and_show()


# Bereczki Bence - Lítiumion-akkumulátoros energiatároló rendszerek alkalmazása ipari környezetben
# http://tdk.bme.hu/VIK/DownloadPaper/Litiumionakkumulatoros-energiatarolo1

# Bereczki pp25
# E-On Mix
demand_charge = 5028 # HUF/kW/year
on_peak_price = 51.12 # HUF/kWh
off_peak_price = 27.48 # HUF/kWh

bess_nominal_capacity = 94 # Ah (Bereczki pp26 Table 4)

# WOWOWOW (everywhere this means some totally unreasonable assumption or unfinished business.)
bess_nominal_capacity = 10000 # totally unrealistic, just to see something meaningful.

voltage = 70.4 # V
maximal_depth_of_discharge = 0.75
bess_capacity = bess_nominal_capacity * voltage / 1000 # kWh. (V*Ah = Wh)
energy_loss = 0.1
# WOWOWOW
bess_charge_per_time_unit = 5

def histogram(c):
    h = np.sort(c)
    plt.plot(h[::10])
    plt.title("Cumulative histogram of consumption, kW")
    save_and_show()

histogram(c)

# we gotta start somewhere
# WOWOWOW
charge_threshold = np.percentile(c, 90)

# TODO port to pandas dataframe.
def simulator(demand_series):
    soc_series = [] # soc = state_of_charge.
    # by convention, we only call end user demand, demand,
    # and we only call end user consumption, consumption.
    # in our simple model, demand is always satisfied, hence demand=consumption.
    # BESS demand is called charge.
    consumption_from_network_series = [] # demand satisfied by network
    consumption_from_bess_series = [] # demand satisfied by BESS
    # the previous two must sum to demand_series.
    charge_of_bess_series = [] # power taken from network by BESS
    funds_spent_series = []
    funds_spent_counterfactually_series = [] # without the peak shaving system.

    soc = 0.0 # we start with an empty battery. 1 is not nominal but targeted (healthy) maximum charge.
    for i, demand in enumerate(demand_series):

        # these three are modified on the appropriate codepaths:
        consumption_from_network = demand
        consumption_from_bess = 0
        charge_of_bess = 0

        if demand > charge_threshold:
            # if we have the battery charged enough:
            # this can go below because of the finite step simualtion,
            # but it's okay like this.
            is_battery_charged_enough = (soc > 1 - maximal_depth_of_discharge)
            if is_battery_charged_enough:
                consumption_from_network = charge_threshold
                # simplifying assumption for now:
                # throughput is enough to completely fulfill extra demand.
                consumption_from_bess = demand - charge_threshold
                bess_sacrifice = consumption_from_bess / (1 - energy_loss)
                soc -= bess_sacrifice / bess_capacity
            else:
                # battery is empty:
                pass
        else:
            # this can go above because of the finite step simualtion,
            # but it's okay like this.
            is_battery_chargeable = (soc < 1.0)
            if is_battery_chargeable:
                charge_of_bess = bess_charge_per_time_unit
                if charge_of_bess + demand > charge_threshold:
                    charge_of_bess = charge_threshold - demand
                    assert charge_of_bess >= 0
                soc += charge_of_bess / bess_capacity
        funds_spent = 0 # WOWOWOW
        funds_spent_counterfactually = 0 # WOWOWOW
        soc_series.append(soc)
        consumption_from_network_series.append(consumption_from_network)
        consumption_from_bess_series.append(consumption_from_bess)
        charge_of_bess_series.append(charge_of_bess)
        funds_spent_series.append(funds_spent)
        funds_spent_counterfactually_series.append(funds_spent_counterfactually)

    soc_series = np.array(soc_series)
    consumption_from_network_series = np.array(consumption_from_network_series)
    consumption_from_bess_series = np.array(consumption_from_bess_series)
    charge_of_bess_series = np.array(charge_of_bess_series)
    funds_spent_series = np.array(funds_spent_series)
    funds_spent_counterfactually_series = np.array(funds_spent_counterfactually_series)

    funds_saved = funds_spent_counterfactually_series.sum() - funds_spent_series.sum()
    n = 2000
    plt.plot(demand_series.to_numpy()[:n], label='Demand')
    plt.plot(consumption_from_network_series[:n], label='Demand served by network')
    plt.plot(consumption_from_bess_series[:n], label='Demand served by BESS')
    plt.plot(charge_of_bess_series[:n], label='Consumption of BESS')
    plt.legend()
    plt.show()
    # save_and_show()
    return funds_saved


print(simulator(c))
