#                              LIBRARY
# ============================================================================================ 
import dash 
from dash import dcc, html, get_relative_path
import dash_leaflet as dl
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_echarts
import functions.maps as FcMap

#                               ACTIVATE MULTIPAGES
# ============================================================================================ 
dash.register_page(__name__, path="/")

#                               GENERAL LAYOUT
# ============================================================================================ 


buildingsLayout = [
    html.Div(
        className = "col-lg-5 col-md-12",
        children = [
            html.Div(
                children = [
                    dcc.Loading(
                        [
                            html.Div(
                                children = [
                                    dl.Map([dl.TileLayer(), FcMap.leaflet_only_buildings()],
                                        #    center=[56.95360157581415, 24.118087122613346],
                                           id="map_1", style={ 'height': '100vh', 'margin': "auto","position": "relative","display": "block", 'zIndex':'50'} ), 
                                ]
                            )
                        ]
                    )
                ]
            ) 
        ],
        style = {'marginTop':'30px'}
    ),
    html.Div(
        className= "col-lg-7 col-md-12",
        children = [
            dmc.Text(
                id = "buiNameSelected",
                children= ["Building: Baldone_1"],
                color = "black",
                # variant="gradient",
                # gradient={"from": "blue", "to": "green", "deg": 45},
                style={
                    'fontSize': 30, 
                    'fontWeight':800,
                    #    'marginBottom':'10px',
                    'marginTop':'40px'}
            ),
            dmc.Group(
                children = [
                    dash_echarts.DashECharts(
                        id = 'gaugeBui',
                        style={
                            "width": '400px',
                            "height": '350px',
                            }
                    ),
                    dash_echarts.DashECharts(
                        id = 'heatBeforeAfter',
                        style={
                            "width": '400px',
                            "height": '350px',
                            }
                    ),
                ]
            ),
            dmc.Group(
                children = [
                    dash_echarts.DashECharts(
                        id = 'costSavingRenovation',
                        style={
                            "width": '400px',
                            "height": '350px',
                            }
                    ),
                    # dmc.Stack(
                    #     children = [
                    #         cardPBT,   
                    #         card_bui_info,                                                
                    #     ]
                    # )
                ],
                grow=True
            ),
        ]
    ),
]


layout_admin = html.Div(
    children = [
        html.Div(
            id = "page-content-home", 
            children = [
                # HeaderAdmin,
                # DrawerRight,
                html.Div(
                    className="row",
                    id = "content_admin_page",
                    children = buildingsLayout
                )
            ]
        )
    ],
    style = {'paddingTop':'45px'}    
)

#                               FINAL LAYOUT
# ============================================================================================ 
layout_admin = html.Div(
    children = [
        html.Div(
            id = "page-content-home", 
            children = [
                # HeaderAdmin,
                # DrawerRight,
                html.Div(
                    className="row",
                    id = "content_admin_page",
                    # children = buildingsLayout
                )
            ]
        )
    ],
    style = {'paddingTop':'45px'}    
)


def layout():
    return layout_admin