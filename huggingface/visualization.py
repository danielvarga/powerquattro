import plotly
import plotly.subplots
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt


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


MARGIN = dict(
    l=0,
    r=0,
    b=0,
    t=0,
    pad=0
)


def plotly_visualize_simulation(results, date_range):
    start_date, end_date = date_range
    results = results.loc[start_date: end_date]
    '''
    fig = px.area(results, x=results.index, y="consumption_from_network")
    return fig'''

    fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(yaxis2=dict(range=[0.0, 110]))

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
    fig.add_trace(go.Scatter(
        x=results.index, y=results['charge_of_bess'] * 100,
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=1.5, color='red'),
        name='State of charge'),
        secondary_y=True
    )
    # could not kill the huge padding this introduces:
    # fig.update_layout(title=f"Simulation for {start_date} - {end_date}")
    fig.update_layout(height=400, yaxis_title="Consumption [kW]", yaxis2_title="State of charge [%]", yaxis2_showgrid=False)

    return fig


def plotly_visualize_monthly(consumption):
    # months = monthly_results.index
    months = list(range(1, 13))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=consumption[:, 0], # monthly_results['consumption_from_network'],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        name='Network',
        stackgroup='one' # define stack group
    ))
    fig.add_trace(go.Scatter(
        x=months, y=consumption[:, 1], # y=monthly_results['consumption_from_solar'],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='orange'),
        name='Solar',
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=months, y=consumption[:, 2], # y=monthly_results['consumption_from_bess'],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='green'),
        name='BESS',
        stackgroup='one'
    ))
    fig.update_layout(
        yaxis_title="Monthly consumption in [MWh]",
        height=400
    )
    return fig


def monthly_visualization(consumptions_in_mwh):
    percentages = consumptions_in_mwh[:, :3] / consumptions_in_mwh.sum(axis=1, keepdims=True) * 100
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
