# port of
# https://colab.research.google.com/drive/1PJgcJ4ly7x5GuZy344eJeYSODo8trbM4#scrollTo=39F2u-4hvwLU

import numpy as np
import pandas as pd
import gradio as gr

from simulation import *
from data_processing import *
from visualization import *


#@title ### Downloading the data
# !wget "https://static.renyi.hu/ai-shared/daniel/pq/PL_44527.19-21.csv.gz"
# !wget "https://static.renyi.hu/ai-shared/daniel/pq/pq_terheles_2021_adatok.tsv"


met_2021_data, cons_2021_data = read_datasets()


def recalculate(**uiParameters):
    fixed_consumption = uiParameters['fixed_consumption']
    del uiParameters['fixed_consumption']

    parameters = Parameters()
    for k, v in uiParameters.items():
        setattr(parameters, k, v)

    add_production_field(met_2021_data, parameters)
    all_2021_data = interpolate_and_join(met_2021_data, cons_2021_data)

    if fixed_consumption:
        all_2021_data['Consumption'] = 10

    results = simulator_with_solar(all_2021_data, parameters)
    return results


def ui_refresh(solar_cell_num, bess_nominal_capacity, fixed_consumption):
    results = recalculate(solar_cell_num=solar_cell_num, bess_nominal_capacity=bess_nominal_capacity, fixed_consumption=fixed_consumption)

    fig1 = plotly_visualize_simulation(results, date_range=("2021-02-01", "2021-02-07"))
    fig2 = plotly_visualize_simulation(results, date_range=("2021-08-02", "2021-08-08"))

    # (12, 3), the 3 indexed with (network, solar, bess):
    consumptions_in_mwh = monthly_analysis(results)

    fig_monthly = plotly_visualize_monthly(consumptions_in_mwh)

    network, solar, bess = consumptions_in_mwh.sum(axis=0)
    html = "<table>\n"
    for column, column_name in zip((network, solar, bess), ("Network", "Solar directly", "Solar via BESS")):
        html += f"<tr><td>Yearly consumption served by {column_name}:&nbsp;&nbsp;&nbsp;</td><td>{column:0.2f} MWh</td></tr>\n"
    html += "</table>"

    return (html, fig_monthly, fig1, fig2)


ui = gr.Interface(
    ui_refresh,
    inputs = [
        gr.Slider(0, 2000, 114, label="Solar cell number"),
        gr.Slider(0, 2000, 330, label="BESS nominal capacity in [Ah]"),
        gr.Checkbox(value=False, label="Use fixed consumption (10 kW)")],
    outputs = ["html", "plot", "plot", "plot"],
    live=True,
)

ui.launch(show_api=False)
