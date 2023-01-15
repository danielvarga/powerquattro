import pandas as pd
import numpy as np

from data_processing import add_production_field, interpolate_and_join, monthly_analysis


def simulator_with_solar(all_data, parameters):
    demand_np = all_data['Consumption'].to_numpy()
    production_np = all_data['Production'].to_numpy()
    assert len(demand_np) == len(production_np)
    step_in_minutes = all_data.index.freq.n
    # print("Simulating for", len(demand_np), "time steps. Each step is", step_in_minutes, "minutes.")
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

    # For the sake of simplicity 0 <= soc <=1
    # soc=0 means battery is emptied till it's 20% and soc=1 means battery is charged till 80% of its capacity
    # soc = 1 - maximal_depth_of_discharge
    # and will use only maximal_depth_of_discharge percent of the real battery capacity
    soc = 0
    max_cap_of_battery = parameters.bess_capacity * parameters.maximal_depth_of_discharge
    cap_of_battery = soc * max_cap_of_battery

    time_interval = step_in_minutes / 60 # amount of time step in hours
    for i, (demand, production) in enumerate(zip(demand_np, production_np)):

        # these five are modified on the appropriate codepaths:
        consumption_from_solar = 0
        consumption_from_bess = 0
        consumption_from_network = 0
        discarded_production = 0

        unsatisfied_demand = demand
        remaining_production = production

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
        
        if parameters.bess_present:
            is_battery_charged_enough = soc > 0
            is_battery_chargeable = soc < 1.0
        else:
            is_battery_charged_enough = soc <= 0
            is_battery_chargeable = soc >= 1.0

        if unsatisfied_demand >= remaining_production:
            # all goes to demand
            consumption_from_solar = remaining_production
            unsatisfied_demand -= consumption_from_solar
            remaining_production = 0
            # we try to cover the rest from BESS
            if (unsatisfied_demand > 0 ) and parameters.bess_present:
                if is_battery_charged_enough:
                    # battery capacity is limited!
                    if cap_of_battery >= unsatisfied_demand * time_interval :
                        consumption_from_bess = unsatisfied_demand
                        unsatisfied_demand = 0
                        cap_of_battery -= consumption_from_bess * time_interval
                        soc = cap_of_battery / max_cap_of_battery
                    else:
                        discharge_of_bess = cap_of_battery / time_interval
                        discharge = min(parameters.bess_discharge, discharge_of_bess)
                        consumption_from_bess = discharge
                        unsatisfied_demand -= consumption_from_bess
                        cap_of_battery -= consumption_from_bess * time_interval
                        soc = cap_of_battery / max_cap_of_battery
                        consumption_from_network = unsatisfied_demand
                        unsatisfied_demand = 0
                else:
                    # we cover the rest from network
                    consumption_from_network = unsatisfied_demand
                    unsatisfied_demand = 0
        else:
            # demand fully satisfied by production
            consumption_from_solar = unsatisfied_demand
            remaining_production -= unsatisfied_demand
            unsatisfied_demand = 0
            if (remaining_production > 0) and parameters.bess_present:
                # exploitable production still remains:
                if is_battery_chargeable:
                    # we try to specify the BESS modell
                    if parameters.bess_charge <= remaining_production :
                        energy = parameters.bess_charge * time_interval
                        remaining_production = remaining_production - parameters.bess_charge
                    else :
                        energy = remaining_production * time_interval
                        remaining_production = 0
                    cap_of_battery += energy
                    soc = cap_of_battery / max_cap_of_battery

        discarded_production = remaining_production

        soc_series.append(soc)
        consumption_from_solar_series.append(consumption_from_solar)
        consumption_from_network_series.append(consumption_from_network)
        consumption_from_bess_series.append(consumption_from_bess)
        charge_of_bess_series.append(soc)
        discarded_production_series.append(discarded_production)

    soc_series = np.array(soc_series)
    consumption_from_solar_series = np.array(consumption_from_solar_series)
    consumption_from_network_series = np.array(consumption_from_network_series)
    consumption_from_bess_series = np.array(consumption_from_bess_series)
    charge_of_bess_series = np.array(charge_of_bess_series)
    discarded_production_series = np.array(discarded_production_series)

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


def evaluate_parameters(parameters, met_2021_data, cons_2021_data):
    add_production_field(met_2021_data, parameters)
    all_2021_data = interpolate_and_join(met_2021_data, cons_2021_data)
    results = simulator_with_solar(all_2021_data, parameters)
    consumptions_in_mwh = monthly_analysis(results)
    return consumptions_in_mwh.sum(axis=0)
