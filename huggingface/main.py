import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm

from simulation import *
from data_processing import *
from visualization import *


matplotlib.rcParams['figure.figsize'] = [12, 8]


def main():
    parameters = Parameters()

    met_2021_data, cons_2021_data = read_datasets()

    add_production_field(met_2021_data, parameters)

    all_2021_data = interpolate_and_join(met_2021_data, cons_2021_data)

    results = simulator_with_solar(all_2021_data, parameters)

    fig = visualize_simulation(results, date_range=("2021-02-01", "2021-03-01"))
    plt.show()

    consumptions_in_mwh = monthly_analysis(results)
    monthly_visualization(consumptions_in_mwh)


# main() ; exit()



def main_gridsearch():
    fixed_consumption = False

    parameters = Parameters()
    met_2021_data, cons_2021_data = read_datasets()

    if fixed_consumption:
        cons_2021_data['Consumption'] = 10

    N = 20
    solar_cell_num_max = 4000
    bess_nominal_capacity_max = 4000
    solar_cell_nums = np.linspace(0, solar_cell_num_max, N)
    bess_nominal_capacities = np.linspace(1e-6, bess_nominal_capacity_max, N)

    mg_x, mg_y = np.meshgrid(solar_cell_nums, bess_nominal_capacities)

    values = np.zeros((N, N))
    for i, solar_cell_num in enumerate(solar_cell_nums):
        print(f"{solar_cell_num} / {solar_cell_nums[-1]}")
        for j, bess_nominal_capacity in enumerate(bess_nominal_capacities):
            parameters.solar_cell_num = solar_cell_num
            parameters.bess_nominal_capacity = bess_nominal_capacity
            network, solar, bess = evaluate_parameters(parameters, met_2021_data, cons_2021_data)
            satisfied = 1 - network / (network + solar + bess)
            values[i, j] = satisfied

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mg_x, mg_y, values * 100, cmap=matplotlib.cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel("BESS nominal capacity [Ah]")
    ax.set_ylabel("Solar cell number")
    ax.set_zlabel("Percentage of consumption served without network")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()


main_gridsearch() ; exit()
