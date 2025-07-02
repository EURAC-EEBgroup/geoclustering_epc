import dash
from dash import html, get_relative_path
import  dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash_mantine_react_table import DashMantineReactTable
import pandas as pd
import dash_echarts
import dash_leaflet as dl
from pages.processing import list_of_lists, accordions, filters, graph_1
from pages.analysis import data_table, plots_univariate_distribution, correlation_matrix, info_1
from pages.clustering import clustering_analysis, prediction

dash.register_page(__name__,path=f"/synthetic_epc")

 
main_synthetic_epc= html.Div(
    children = [
        dmc.AppShell(
            [
                dmc.AppShellMain(
                    children=[
                        dmc.Container(
                            size="xl",
                            children = [
                                 html.Iframe(
                                    id="iframe_report",
                                    src = get_relative_path("/assets/census-tabular.html"),
                                    style={"width": "100%", "height": "500vh", "border": "none"}
                                ), 
                            ],
                            style = {'scrollBehavior': "smooth"}
                        )
                        ],
                    p=30
                ),
                
            ]
        ),
    ],
    style = {'marginBottom':'20px'}
)

def layout():
    return main_synthetic_epc


