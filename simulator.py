# port of
# https://colab.research.google.com/drive/1PJgcJ4ly7x5GuZy344eJeYSODo8trbM4#scrollTo=39F2u-4hvwLU

from dataclasses import dataclass
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import datetime
from scipy.interpolate import interp1d   

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go


#@title ### Downloading the data
# !wget "https://static.renyi.hu/ai-shared/daniel/pq/PL_44527.19-21.csv"
# !wget "https://static.renyi.hu/ai-shared/daniel/pq/pq_terheles_2021_adatok.tsv"


PATH_PREFIX = "./"

matplotlib.rcParams['figure.figsize'] = [12, 8]

START = f"2021-01-01"
END = f"2022-01-01"


def read_datasets():
    #@title ### Preprocessing meteorologic data
    met_data = pd.read_csv(PATH_PREFIX + 'PL_44527.19-21.csv', sep=';', skipinitialspace=True, na_values='n/a', skiprows=[0, 1, 2, 3, 4])
    met_data['Time'] = met_data['Time'].astype(str)
    date_time = met_data['Time'] = pd.to_datetime(met_data['Time'], format='%Y%m%d%H%M')
    met_data = met_data.set_index('Time')


    #@title ### Preprocessing consumption data
    cons_data = pd.read_csv(PATH_PREFIX + 'pq_terheles_2021_adatok.tsv', sep='\t', skipinitialspace=True, na_values='n/a', decimal=',')
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


def simulator_with_solar(all_data, parameters):
    
    demand_np = all_data['Consumption'].to_numpy()
    production_np = all_data['Production'].to_numpy()
    assert len(demand_np) == len(production_np)
    step_in_minutes = all_data.index.freq.n
    print("Simulating for", len(demand_np), "time steps. Each step is", step_in_minutes, "minutes.")
    soc_series = [] # soc = state_of_charge.
    # by convention, we only call end user demand, demand,
    # and we only call end user consumption, consumption.
    # in our simple model, demand is always satisfied, hence demand=consumption.
    # BESS demand is called charge.
    consumption_from_solar_series = [] # demand satisfied by solar production
    consumption_from_network_series = [] # demand satisfied by network
    consumption_from_bess_series = [] # demand satisfied by BESS
    # the previous three must sum to demand_series.
    charge_of_bess_series = [] # power taken from solar by BESS. note: power never taken from network by BESS.
    discarded_production_series = [] # solar power thrown away

    # 1 is not nominal but targeted (healthy) maximum charge.
    # we start with an empty battery, but not emptier than what's healthy for the batteries.
    
    #Remark from Jutka
    #For the sake of simplicity 0<= soc <=1
    #soc = 1 - maximal_depth_of_discharge
    #and will use only  maximal_depth_of_discharge percent of the real battery capacity
    soc = 0
    max_cap_of_battery = parameters.bess_capacity * parameters.maximal_depth_of_discharge
    cap_of_battery = soc * max_cap_of_battery

    time_interval = step_in_minutes / 60 # amount of time step in hours
    for i, (demand, production) in enumerate(zip(demand_np, production_np)):

        # these three are modified on the appropriate codepaths:
        consumption_from_solar = 0
        consumption_from_bess = 0
        consumption_from_network = 0
        charge_of_bess = 0
        
        #Remark: If the consumption stable for ex. 10 kwh
        # demand = 10
        unsatisfied_demand = demand
        remaining_production = production # max((production, 0))
        discarded_production = 0
        
        # crucially, we never charge the BESS from the network.
        # if demand >= production:
        #   all goes to demand
        #   we try to cover the rest from BESS
        #   we cover the rest from network
        # else:
        #   demand fully satisfied by production
        #   if exploitable production still remains:
        #     if is_battery_chargeable:
        #       charge_battery
        #     else:
        #       log discarded production

        #battery_charged_enough = (soc > 1- maximal_depth_of_discharge)
        is_battery_charged_enough = (soc > 0 )
        is_battery_chargeable = (soc < 1.0)

        if unsatisfied_demand >= remaining_production:
        #   all goes to demand
            consumption_from_solar = remaining_production
            unsatisfied_demand -= consumption_from_solar
            remaining_production=0 #edited by Jutka
        #   we try to cover the rest from BESS
            
            if unsatisfied_demand > 0:
                if is_battery_charged_enough:
                    # simplifying assumption for now:
                    # throughput is enough to completely fulfill extra demand.
                    # TODO get rid of simplifying assumption.
                    # Remarks from Jutka
                    # It is a very bed assumption. The reality needs, that the BESS has limited capacity. 
                    #
                    #                   
                    # cap_of_bess=soc * bess_capacity
                    # if cap_of_bess > unsatisfied_demand
                    #     consumption_from_bess = unsatisfied_demand
                    #     unsatisfied_demand = 0
                    #     cap_of_bess -= consumption_from_bess
                    #     soc = cap_of_bess / bess_capacity
                    # else: unsatisfied_demand -= cap_of_bess
                    #       cap_of_bess = 0
                    #       soc = 0
                    #       if unsatisfied_demand > 0
                    #          consumption_from_network = unsatisfied_demand
                    #          unsatisfied_demand = 0
                  
                    #Remarks: battery capacity is limited!
                   
                     if cap_of_battery >= unsatisfied_demand * time_interval :
                     
                       #discharge_of_bess = min ( unsatisfied_demand, bess_discharge )
                       #discharge = discharge_of_bess
                       #consumption_from_bess = discharge * time_interval
                       consumption_from_bess = unsatisfied_demand 
                       #unsatisfied_demand -= consumption_from_bess
                       unsatisfied_demand = 0
                       cap_of_battery -= consumption_from_bess * time_interval
                       soc = cap_of_battery / max_cap_of_battery
                       
                     else:
                      #discharge_of_bess = cap_of_battery /time_interval
                      #discharge = min( bess_discharge, discharge_of_bess )
                      consumption_from_bess = cap_of_battery / time_interval
                      unsatisfied_demand -= consumption_from_bess
                      cap_of_battery -=consumption_from_bess * time_interval
                      soc = cap_of_battery / max_cap_of_battery
                      consumption_from_network = unsatisfied_demand
                      unsatisfied_demand = 0 
                    #bess_sacrifice = consumption_from_bess / (1 - energy_loss) # kW
                    #energy = bess_sacrifice * time_interval # kWh
                    #soc -= energy / bess_capacity
                    # print("soc after discharge", soc)
                    #consumption_from_network = unsatisfied_demand
                    #unsatisfied_demand = 0 
                else:
                    #   we cover the rest from network
                  consumption_from_network = unsatisfied_demand
                  unsatisfied_demand = 0
        
        else:
        #   demand fully satisfied by production
          
            consumption_from_solar = unsatisfied_demand
            remaining_production -= unsatisfied_demand
            unsatisfied_demand = 0
        #   if exploitable production still remains:
            if remaining_production > 0:
                if is_battery_chargeable:
                    charge_of_bess =  remaining_production
                    energy = charge_of_bess * time_interval # kWh
                    #Remarks: battery alowed to charge until its capacity maximum
                    #energy_charge = min(energy, max_cap_of_battery-cap_of_battery)
                    cap_of_battery += energy
                    #soc += energy / bess_capacity
                    soc = cap_of_battery / max_cap_of_battery
                    #print("soc after charge", soc)
                else:
                    discarded_production = remaining_production

        soc_series.append(soc)
        consumption_from_solar_series.append(consumption_from_solar)
        consumption_from_network_series.append(consumption_from_network)
        consumption_from_bess_series.append(consumption_from_bess)
        charge_of_bess_series.append(charge_of_bess)
        discarded_production_series.append(discarded_production)

    soc_series = np.array(soc_series)
    consumption_from_solar_series = np.array(consumption_from_solar_series)
    consumption_from_network_series = np.array(consumption_from_network_series)
    consumption_from_bess_series = np.array(consumption_from_bess_series)
    charge_of_bess_series = np.array(charge_of_bess_series)
    discarded_production_series = np.array(discarded_production)

    results = pd.DataFrame({'soc_series': soc_series, 'consumption_from_solar': consumption_from_solar_series,
                            'consumption_from_network': consumption_from_network_series,
                            'consumption_from_bess': consumption_from_bess_series,
                            'charge_of_bess': charge_of_bess_series,
                            'discarded_production': discarded_production_series,
                            'Consumption': all_data['Consumption'],
                            'Production': all_data['Production']
                            })
    results = results.set_index(all_data.index)
    return results


def visualize_simulation(results, date_range):
    start_date, end_date = date_range

    fig = plt.figure()
    results = results.loc[start_date: end_date]

    x = results.index
    y = [results.consumption_from_solar, results.consumption_from_network, results.consumption_from_bess]
    plt.plot(x, y[0], label='Demand served by solar', color='yellow', linewidth=0.5)
    plt.plot(x, y[0]+y[1], label='Demand served by network', color='blue', linewidth=0.5)
    plt.plot(x, y[0]+y[1]+y[2], label='Demand served by BESS', color='green', linewidth=0.5)
    plt.fill_between(x, y[0]+y[1]+y[2], 0, color='green')
    plt.fill_between(x, y[0]+y[1], 0, color='blue')
    plt.fill_between(x, y[0], 0, color='yellow')

    # plt.xlim(datetime.datetime.fromisoformat(start_date), datetime.datetime.fromisoformat(end_date))

    plt.legend()
    return fig


def plotly_visualize_simulation(results, date_range):
    start_date, end_date = date_range
    results = results.loc[start_date: end_date]
    '''
    fig = px.area(results, x=results.index, y="consumption_from_network")
    return fig'''
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results['consumption_from_network'],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        name='Network',
        stackgroup='one' # define stack group
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results['consumption_from_solar'],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='orange'),
        name='Solar',
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results['consumption_from_bess'],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='green'),
        name='BESS',
        stackgroup='one'
    ))
    fig.update_layout(
        height=400
    )
    return fig


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
    percentages = consumptions[:, :3] / consumptions.sum(axis=1, keepdims=True) * 100
    bats = 0
    nws = 0
    sols = 0

    print("[Mwh]")
    print("==========================")
    print("month\tnetwork\tsolar\tbess")
    for month_minus_1 in range(12):
        network, solar, bess = consumptions_in_mwh[month_minus_1]
        print(f"{month_minus_1+1}\t{network:0.2f}\t{solar:0.2f}\t{bess:0.2f}")
        bats += bess
        nws += network
        sols += solar
    print(f"\t{nws:0.2f}\t{sols:0.2f}\t{bats:0.2f}")


    fig, ax = plt.subplots()

    ax.stackplot(range(1, 13),
                  percentages[:, 0], percentages[:, 1], percentages[:, 2],
                  labels=["hálózat", "egyenesen a naptól", "a naptól a BESS-en keresztül"])
    ax.set_ylim(0, 100)
    ax.legend()
    plt.title('A fogyasztás hány százalékát fedezte az adott hónapban?')
    plt.show()

    plt.stackplot(range(1, 13),
                  consumptions_in_mwh[:, 0], consumptions_in_mwh[:, 1], consumptions_in_mwh[:, 2],
                  labels=["hálózat", "egyenesen a naptól", "a naptól a BESS-en keresztül"])
    plt.legend()
    plt.title('Mennyi fogyasztást fedezett az adott hónapban? [MWh]')
    plt.show()



def main():
    parameters = Parameters()

    met_2021_data, cons_2021_data = read_datasets()

    add_production_field(met_2021_data, parameters)

    all_2021_data = interpolate_and_join(met_2021_data, cons_2021_data)

    results = simulator_with_solar(all_2021_data, parameters)

    fig = visualize_simulation(results, date_range=("2021-02-01", "2021-03-01"))
    plt.show()

    monthly_analysis(results)


met_2021_data, cons_2021_data = read_datasets()


def recalculate(**uiParameters):
    parameters = Parameters()
    for k, v in uiParameters.items():
        setattr(parameters, k, v)

    add_production_field(met_2021_data, parameters)
    all_2021_data = interpolate_and_join(met_2021_data, cons_2021_data)
    results = simulator_with_solar(all_2021_data, parameters)
    return results


def ui_refresh(solar_cell_num, bess_nominal_capacity):
    results = recalculate(solar_cell_num=solar_cell_num, bess_nominal_capacity=bess_nominal_capacity)

    fig1 = plotly_visualize_simulation(results, date_range=("2021-02-01", "2021-02-07"))
    fig2 = plotly_visualize_simulation(results, date_range=("2021-08-02", "2021-08-08"))

    return (fig1, fig2)


ui = gr.Interface(
    ui_refresh,
    inputs = [
        gr.Slider(0, 1000, 114, label="Solar cell number"),
        gr.Slider(0, 1000, 330, label="BESS nominal capacity")],
    outputs = ["plot", "plot"],
    live=True,
)

ui.launch()
