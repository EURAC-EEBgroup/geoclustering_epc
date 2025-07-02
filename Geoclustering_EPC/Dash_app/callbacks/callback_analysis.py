from dash import callback, Input, Output, State, ALL, html, clientside_callback, ClientsideFunction, get_relative_path
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from globals import df
import dash_leaflet as dl
import dash_leaflet.express as dlx
from utils.functions import create_map


# ==============================================================================================================================
#                                   DEFINITION CONTENT SINGLE ARGUMENTS   
# ==============================================================================================================================

clientside_callback(
    ClientsideFunction("clientside", "project_navBtnClick"),
    Output('clientside_callback_output', 'children', allow_duplicate=True),
    Input('btnAnalysis', 'n_clicks'),
    State("prj_card_analysis", "id"),
    prevent_initial_call='initial_duplicate'
)

clientside_callback(
    ClientsideFunction("clientside", "project_navBtnClick"),
    Output('clientside_callback_output', 'children', allow_duplicate=True),
    Input('btnProcessing', 'n_clicks'),
    State("prj_card_processing", "id"),
    prevent_initial_call='initial_duplicate'
)

clientside_callback(
    ClientsideFunction("clientside", "project_navBtnClick"),
    Output('clientside_callback_output', 'children', allow_duplicate=True),
    Input('btnClustering', 'n_clicks'),
    State("prj_card_clustering", "id"),
    prevent_initial_call='initial_duplicate'
)

clientside_callback(
    ClientsideFunction("clientside", "project_onscroll"),
    output = Output('clientside_callback_output', 'children'),
    inputs = [Input('prj_card_analysis', 'id')],
)

@callback(
    Output("btnClustering", "disabled"),
    Output("btnProcessing", "disabled"),
    Output("btnAnalysis", "disabled"),
    Input("url_app", "pathname"),
)
def disable_btnClustering(pathname):
    if pathname == '/epc_clustering/piemonte/synthetic_epc':
        return True, True, True
    else:
        return False, False, False

# ================================================================
#                   GRAPH UNIVARIATE DITRIBUTION
# ================================================================
@callback(
    Output("univariate_dist_areachart", "data"),
    Input("parameters", "value"),
)
def plot_distribution_graph(parameter):
    '''
    Plot univariate distribution graph
    '''
    data = df[parameter]

    # Compute density estimation
    x_values = np.linspace(data.min(), data.max(), 100)
    kde = gaussian_kde(data)
    y_values = kde(x_values)
    df_y_values = pd.DataFrame(y_values).round(4)
    df_y_values.columns = ['values']
    data = df_y_values.to_dict(orient="records")

    return data

# ================================================================
#                   VISUALIZE INFORMATION IN THE MAP
# ================================================================
'''
type of Visualization:
1. by building typology
2. construction year:
    before 1900
'''

def categorize_year(year):
    """Classifies a given year into one of six categories."""
    if year < 1900:
        return 1 #"Before 1900"
    elif 1900 <= year < 1950:
        return 2 #"1900-1949"
    elif 1950 <= year < 1970:
        return 3 #"1950-1969"
    elif 1970 <= year < 1980:
        return 4 #"1970-1979"
    elif 1980 <= year < 1990:
        return 5 #"1980-1989"
    elif 1990 <= year < 2000:
        return 6 #"1990-1999"
    elif 2000 <= year < 2010:
        return 7 #"2000-2010"
    else:
        return 8 #"2010-Present"



@callback(
    Output("map_bui", "children"),
    Input("map_inputs", "value")
)
def visualize_building_in_map(map_inputs):
    '''
    Visualize building in map
    '''
    data_map = df.loc[:, ['latitude', 'longitude']]
    if map_inputs == "DPR412_classification":
        data_map['DPR412_classification'] = df.loc[:, ['DPR412_classification']]
        data_map.columns = ['lat', 'lon', 'variable']
    elif map_inputs == "construction_year":
        categorized_years = [categorize_year(year) for year in df['construction_year']]
        data_map['categorized_year'] = categorized_years
        data_map.columns = ['lat', 'lon', 'variable']
    
    return create_map(data_map, id_map="map_bui")

# ================================================================
#                   CORRELATION MATRIX
# ================================================================

@callback(
    Output("correlation_heat_chart", "option"),
    Input("url_app", "href"),
)
def correlation_matrix(href_):
    '''
    Visualize correlation matrix of all parameters
    '''

    
    # df_ = df.drop('comune_catastale', axis=1)
    # df_ = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
    # df__ = df_[~df_.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]

    # Compute correlation matrix
    corr_matrix = df.corr()
    columns = corr_matrix.columns.tolist()
    data = []
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            data.append([i, j, round(corr_matrix.loc[col1, col2], 2)])

    option={
            "tooltip": {"position": "top"},
            # "grid": {"height": "60%", "top": "10%"},
            "xAxis": {"type": "category", "data": columns},
            "yAxis": {"type": "category", "data": columns},
            "visualMap": {"min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center"},
            "series": [{
                "name": "Correlation", "type": "heatmap",
                "data": data, 
                # "label": {"show": True}
            }]
        }

    return option



