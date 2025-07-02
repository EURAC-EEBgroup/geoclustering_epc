from dash import callback, Input, Output, State,ctx, get_relative_path,html, dcc
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from pages.analysis import df
from utils.analysis_clustering import elbow_silhouette_method
from utils.functions import create_map
import dash 
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from globals import path_result

@callback(
    Output("cluster_method", "disabled"),
    Output("custom_number", "disabled"),
    Input("cluster_type", "value")
)
def cluster_type_selection(cluster_type):
    if cluster_type == "best_option":
        return False, True
    else:
        return True, False


def evaluate_energy(df_, colName, cluster_number):

    df = df_.groupby("cluster")[colName].mean().reset_index()
    # Transform DataFrame to list of dictionaries with rounded EPh values
    data = []
    for _, row in df.iterrows():
        data.append({
            'cluster': row['cluster'],
            'EPh': round(row['EPh'], 2)
        })

    if cluster_number <= 2:
        energy_evaluation = dmc.Fieldset(
            children = [
                dmc.BarChart(
                h=300,
                dataKey="cluster",
                data=data,
                series=[
                    {"name": "EPh", "color": "violet.6"}
                ],
                tickLine="y",
                gridAxis="y",
                withXAxis=True,
                withYAxis=True
            )
            ],
            legend="Energy EPh"
        )
    else:
        energy_evaluation = dmc.Fieldset(
            children=[
                dmc.RadarChart(
                    h=300,
                    data=data,
                    dataKey="cluster",
                    withPolarRadiusAxis=True,
                    series=[{"name": "EPh", "color": "blue.4", "opacity": 0.2}],
                )
            ],
            legend="Energy EPh"
        )

    overall_layout = html.Div(
        children = [
            dmc.Divider(variant="dotted", mt=20, mb=10, size="md", c="grey"),
            dmc.Title("Analysis of clusters", order=1, mb=10, mt=10),
            dmc.Grid(
                children = [
                    dmc.GridCol(
                        children = [
                            energy_evaluation
                        ],
                        span={"base": 12, "md": 6, "lg":6}
                    ),
                    dmc.GridCol(
                        children = [
                            dmc.Grid(
                                children = [
                                    dmc.GridCol(
                                        children = [
                                            dmc.Select(
                                                id="cluster_list_for_analysis",
                                                label="Cluster",
                                                mt=10,
                                            ),
                                        ],
                                        span={"base": 12, "md": 6, "lg":6}
                                    ),
                                    dmc.GridCol(    
                                        children = [
                                            dmc.Select(
                                                id="parameters_cluster_analysis",
                                                label = "Parameters for clustering",
                                                data = [
                                                    {"label": "Heating energy need - QHnd", "value": "QHnd"},
                                                    {"label": "Primary energy - kwh/m2 year", "value": "EPh"},
                                                    {"label": "Degree days", "value": "degree_days"},
                                                    {"label": "Heat loss surface", "value": "heat_loss_surface"},
                                                    {"label": "Heated usable area", "value": "heated_usable_area"},
                                                    {"label": "Average opaque surface transmittance", "value": "average_opaque_surface_transmittance"},
                                                    {"label": "Average glazed surface transmittance", "value": "average_glazed_surface_transmittance"},
                                                    {"label": "Surface to volume ratio", "value": "surface_to_volume_ratio"},
                                                    {"label": "Exposed solar surface", "value": "Asol"},
                                                    {"label": "Average transmittance of facade", "value": "average_U_facade"},
                                                ],
                                                value="QHnd",
                                                mt=10,
                                            ),
                                        ],
                                        span={"base": 12, "md": 6, "lg":6}
                                    ),
                                ]
                            ),
                            html.Div(id="cluster_selected_distribution"),
                        ],
                        span={"base": 12, "md": 6, "lg":6}
                    )
                ]
            )
        ]
    )
    return overall_layout

@callback(
    Output("map_clustering", "children"),
    Output("elbow_graph", "children"),
    Output("silhouette_graph", "children"),
    Output("data_clustered", "data"),
    Output("cluster_analysis_graphs", "children"),
    Output("number_of_cluster", "data"),
    Input("cluster_btn", "n_clicks"),
    State("cluster_type", "value"),
    State("cluster_method", "value"),
    State("custom_number", "value"),
    State("parameters_cluster", "value"),
    State("building_type", "value"),
    State("Elbow_method", "checked"),
    State("Silhouette_method", "checked"),
    State("data_filtered", "data"),
    # prevent_initial_call=True
)
def visualize_building_in_map(
    cluster_btn, 
    cluster_type, 
    cluster_method_, 
    custom_number, 
    columns_selected, 
    map_inputs, 
    elbow_method, 
    silhouette_method,
    df_filtered):
    '''
    Visualize building in map
    '''
    if ctx.triggered_id == "cluster_btn":
        if cluster_type == "custom_number":
            cluster_value = custom_number
            cluster_method_custom = True
            cluster_method = "elbow"
        else: # Best option
            cluster_method_custom = False
            cluster_value = None
            cluster_method = cluster_method_
        
        if df_filtered:
            df = pd.DataFrame(df_filtered)
        else:
            df = df

        elbow_method_graph,silhouette_method_graph,_,_,df_, optimal_k = elbow_silhouette_method(df, columns_selected, cluster_method_custom, cluster_value, cluster_method)
        if elbow_method:
            chart_elbow = elbow_method_graph
        # else:
        #     chart_elbow = dash.no_update
        if silhouette_method:
            chart_silhouette = silhouette_method_graph
        # else:
        #     chart_silhouette = dash.no_update

        data_map = pd.DataFrame(
            {'lat': df_['latitude'], 'lon': df_['longitude'], 'variable': df_['cluster']}
        )
        data_map['variable'] = data_map['variable'].astype(str).tolist()
        # Evaluate energy graphs
        colName = "EPh"
        evaluate_graphs = evaluate_energy(df_, colName, optimal_k)
    else:
        df = pd.read_csv("Dash_app/data/filtered_data.csv", sep=",", decimal=".", low_memory=False, header=0)
        data_map = df.loc[:, ['latitude', 'longitude', 'DPR412_classification']]
        data_map = data_map.loc[data_map['DPR412_classification'] == int(map_inputs)]
        data_map.columns = ['lat', 'lon', 'variable']
        chart_elbow = dash.no_update
        chart_silhouette = dash.no_update
        df_=pd.DataFrame()
        evaluate_graphs = dash.no_update
        optimal_k = None
    
    return create_map(data_map, "70vh", id_map="map_clustering"), chart_elbow, chart_silhouette,df_.to_dict(),evaluate_graphs, optimal_k

@callback(
    Output("cluster_list_for_analysis", "data"),
    Output("cluster_list_for_analysis", "value"),
    Input("number_of_cluster", "data"),
)
def update_cluster_select(number_of_cluster):
    if number_of_cluster is None:
        return dash.no_update
    return [{"label": f"Cluster {i}", "value": str(i)} for i in range(number_of_cluster)], "0"


@callback(
    Output("cluster_selected_distribution", "children"),
    Input("cluster_list_for_analysis", "value"),
    Input("parameters_cluster_analysis", "value"),
    State("data_clustered", "data"),
)
def update_cluster_selected_distribution(cluster_list_for_analysis, parameters_cluster_analysis, data_clustered):
    if data_clustered:
        df_ = pd.DataFrame(data_clustered)
        df = df_.loc[df_['cluster'] == cluster_list_for_analysis]
        data_ = df.reset_index().sort_values(by=parameters_cluster_analysis, ascending=True).reset_index(drop=True).to_dict(orient="records")
        min = df[parameters_cluster_analysis].values.min().tolist()
        max = df[parameters_cluster_analysis].values.max().tolist()
        mean_ = df[parameters_cluster_analysis].values.mean().tolist()
        ###
        # GRAPH 1
        graph1 = dmc.BarChart(
            dataKey = "index",
            data=data_,
            yAxisLabel=parameters_cluster_analysis,
            series=[
                {"name": parameters_cluster_analysis, "color": "violet.6"}
            ],
            tickLine="y",
            gridAxis="y",
            withXAxis=True,
            withYAxis=True,
            referenceLines=[
                {"y": min, "label": f"Min: {min}", "color": "red.6"},
                {"y": max, "label": f"Max: {max}", "color": "blue.6"},
                {"y": mean_, "label": f"Mean: {mean_}", "color": "green.6"},
       
            ],
            h=250,
            mt=10
        ) 
        return graph1
    else:
        return dash.no_update
# ============================================================
#           PREDICTION GRAPHS ANALYSIS
# ============================================================
# Attiva solo se cluster anaylsi viene eseguito
@callback(
    Output("predict_btn","disabled"),
    Output("cluster_select","disabled"),
    Output("alert_prediction","hidden"),
    Input("cluster_btn", "n_clicks"),
)
def prediction_btn(cluster_btn):
    if ctx.triggered_id == "cluster_btn":
        return False, False, True
    else:
        return True, True, False

# Aggiornare il select con gli effetivi cluster trovati
@callback(
    Output("cluster_select", "data"),
    Output("cluster_select", "value"),
    Input("number_of_cluster", "data"),
)
def update_cluster_select(number_of_cluster):
    if number_of_cluster is None:
        return dash.no_update
    return [{"label": f"Cluster {i}", "value": str(i)} for i in range(number_of_cluster)], "0"


# Map single cluster selected 
@callback(
    Output("map_prediction", "children"),
    Input("cluster_select", "value"),
    State("data_clustered", "data"),
    State("building_type", "value")
)
def prediction_analysis_and_map(cluster_select, data_clustered, building_type):
    df_ = pd.DataFrame(data_clustered)
    if df_.empty:
        data_map = df.loc[:, ['latitude', 'longitude', 'DPR412_classification']]
        data_map = data_map.loc[data_map['DPR412_classification'] == int(building_type)]
        data_map.columns = ['lat', 'lon', 'variable']
    else:
        df_cluster = df_.loc[df_['cluster'] == cluster_select]
        data_map = pd.DataFrame(
            {'lat': df_cluster['latitude'], 'lon': df_cluster['longitude'], 'variable': df_cluster['cluster']}
            )
        data_map['variable'] = data_map['variable'].astype(str).tolist()
    return create_map(data_map, "70vh", id_map="map_prediction")


@callback(
    Output("scenarios_graphs", "children"),
    Output("heat_map_influence", "children"),
    Output("3d_influence", "children"),
    Input("predict_btn", "n_clicks"),
    State("cluster_select", "value")
)
def prediction_analysis_and_map(predict_btn, cluster_select):
    if ctx.triggered_id == "predict_btn":
       
        cluster_data = f"{path_result}/df_confronto_scenari_cluster_{cluster_select}.csv"
        influence_data = f"{path_result}/results_analisi_sensibilita_cluster_{cluster_select}.csv"
        df_influence = pd.read_csv(influence_data)
        df_cluster = pd.read_csv(cluster_data)
        df_scenario = df_cluster.loc[:,['scenario','predizione', 'variazione_pct']]
        df_scenario.columns = ["scenario", "prediction", "variation[%]"]
        df_scenario = df_scenario.to_dict(orient="records")
        for d in df_scenario:
            d['prediction'] = round(d['prediction'], 2)
            d['variation[%]'] = round(d['variation[%]'], 2)
        # BAR CHART

        title = dmc.Stack(
            children = [
                dmc.Title("Scenarios", order=1),
                dmc.Title("Evalute the impact of the selected parameters", order=3, c="#dee2e6")
            ],
            gap="sm",
            mb=10
        )
        
        chart = dmc.BarChart(
            h=300,
            dataKey="scenario",
            data=df_scenario,
            withBarValueLabel=True,
            series=[
                {"name": "prediction", "color": "blue.6", "stackId": "b"},
            ],
            mt=10
        )
        # Table
        rows = [
            dmc.TableTr(
                [
                    dmc.TableTd(element["scenario"]),
                    dmc.TableTd(element["prediction"]),
                    dmc.TableTd(element["variation[%]"]),
                ]
            )
            for element in df_scenario
        ]
        head = dmc.TableThead(
            dmc.TableTr(
                [
                    dmc.TableTh("Scenario"),
                    dmc.TableTh("Prediction"),
                    dmc.TableTh("Variation[%]"),
                ]
            )
        )
        body = dmc.TableTbody(rows)
        caption = dmc.TableCaption("Scenario Evaluation")
        table = dmc.Table([head, body, caption], mt=10, mb=10)

        #
        


        # influenza parameteri: 
        
       
        param1 = "average_opaque_surface_transmittance"
        param2 = "average_glazed_surface_transmittance"
        target = "predizione"
        # Valori degli assi
        x = df_influence[param1].values
        y = df_influence[param2].values
        z = df_influence[target].values

        # Crea la figura con trisurf (equivalente a matplotlib's plot_trisurf)
        fig = go.Figure(
            data=[go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color='lightblue',
                opacity=0.7,
                intensity=z,
                colorscale='Viridis',
                showscale=True
            )]
        )

        
        # Aggiorna layout
        fig.update_layout(
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=target
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        # 
        # Crea tabella pivot (come in seaborn)
        pivot_table = df_influence.pivot_table(
            index=param1, 
            columns=param2, 
            values='predizione'
        )

        # Usa plotly per creare heatmap
        fig1 = px.imshow(
            pivot_table.values,
            labels=dict(x=param2, y=param1, color=target),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale='viridis',
            aspect='auto'
        )
        fig1.update_layout(
            xaxis_title=param2,
            yaxis_title=param1,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        graph_3d = dmc.Stack(
            children = [
                dmc.Title("Heatmap: Evaluate the effect of selected parameters to the target", order=3, mb=10),
                dcc.Graph(id="influence_parameters_graph", figure = fig1)
            ],
            gap="md"
        )

        graph_heatmap = dmc.Stack(
            children = [
                dmc.Title("3D response surface: Evaluate the effect of selected parameters to the target", order=3, mb=10),
                dcc.Graph(id="influence_parameters_graph_heat", figure = fig)
            ],
            gap="md"
        )

        return [title, chart, table], graph_3d, graph_heatmap
    else:
        return [], [], []
