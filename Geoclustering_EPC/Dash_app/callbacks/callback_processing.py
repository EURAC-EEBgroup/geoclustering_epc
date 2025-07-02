from dash import Input, Output, State, ALL,callback,ctx, get_relative_path
import dash_mantine_components as dmc
import utils.functions as Fc
import pandas as pd
from globals import df
# Da muovere in database


# ================================================================
#                   ACTIVATE ALL FILTERS
# ================================================================
@callback(
    Output("all_filters", "checked"),
    Output("all_filters", "indeterminate"),
    Output({"type": "notification-item", "index": ALL}, "checked"),
    Input("all_filters", "checked"),
    Input({"type": "notification-item", "index": ALL}, "checked"),
    prevent_initial_callback=True
)
def update_main_checkbox(all_checked, checked_states):
    # handle "all" checkbox"
    if ctx.triggered_id == 'all_filters':
        checked_states = [all_checked] * len(checked_states)

    # handled individual check boxes
    all_checked_states = all(checked_states)
    indeterminate = any(checked_states) and not all_checked_states
    return all_checked_states, indeterminate, checked_states


# ================================================================
#                   FILTERS
# ================================================================

@callback(
    Output("data_filtered", "data"),
    Output("number_of_epc", "children"),
    Input("id_variable", "value"), 
    Input({"type":"notification-item", "index":0}, "checked"), # FILTER 1
    Input({"type":"notification-item", "index":1}, "checked"), # FILTER 2
    Input({"type":"notification-item", "index":2}, "checked"), # FILTER 3
    Input({"type":"notification-item", "index":3}, "checked"), # FILTER 6
    Input({"type":"notification-item", "index":4}, "checked"), # FILTER 7
    Input({"type":"notification-item", "index":5}, "checked"), # FILTER 8
    Input({"type":"notification-item", "index":6}, "checked"), # FILTER 9
    Input({"type":"notification-item", "index":7}, "checked"), # FILTER 10
    Input({"type":"notification-item", "index":8}, "checked"), # FILTER 11
    Input({"type":"notification-item", "index":9}, "checked"), # FILTER 12
    Input({"type":"notification-item", "index":10}, "checked") # FILTER 13
)
def apply_filters(
    parameter_graph, interfloor, system_efficiency, prim_therm_need, building_geometry, aeroilluminat_ratio, 
    window_transmittance, generator_power, exp_sol_surface, construction_year, min_heated_surface, air_change
    ):
    '''
    Filter dataset according to specifici filters selected
    '''
    df_filtered = Fc.filtering_dataset(
        df, interfloor, system_efficiency, prim_therm_need, True, True, building_geometry, 
        aeroilluminat_ratio, window_transmittance, generator_power, exp_sol_surface, 
        construction_year, min_heated_surface, air_change)
    # df_filtered.to_csv(get_relative_path("/data/data_filtered.csv"))

    df_loc = df_filtered.loc[:, ['DPR412_classification',parameter_graph]]
    epcs = len(df_filtered)
    return df_filtered.to_dict(orient="records"), epcs
    
    

@callback(
    Output("graph_processing_1","children"),
    Input("data_filtered", "data"), 
    Input("id_variable", "value"), 
)
def apply_filters(data, parameter_graph):
    '''
    Udpate dataset based on the checked filters
    '''
    df_loc =pd.DataFrame(data)
    data_ = df_loc.sort_values(by=parameter_graph, ascending=True).reset_index(drop=True).to_dict(orient="records")
    ###
    # GRAPH 1
    graph1 = dmc.BarChart(
        h=350,
        dataKey="DPR412_classification",
        data=data_,
        yAxisLabel=parameter_graph,
        xAxisLabel="DPR412_classification",
        series=[
            {"name": parameter_graph, "color": "violet.6"}
        ],
        tickLine="y",
        gridAxis="y",
        withXAxis=True,
        withYAxis=True
    ) 
    # GRAPH 2
    data_grouped= df_loc.groupby("DPR412_classification").agg({parameter_graph: "mean"}).reset_index().to_dict(orient="records")
    graph2 = dmc.RadarChart(
        h=350,
        data=data_grouped,
        dataKey="DPR412_classification",
        withPolarRadiusAxis=True,
        series=[{"name": parameter_graph, "color": "blue.4", "opacity": 0.2, "strokeColor": "blue"}],
    )
    return [graph1, dmc.Title(f"Distribution of {parameter_graph}", order=4, c="violet"), graph2]


