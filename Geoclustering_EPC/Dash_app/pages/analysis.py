import dash
from dash import html, dcc
import  dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash_mantine_react_table import DashMantineReactTable
import pandas as pd
import dash_echarts
from pages.processing import list_of_lists
from globals import df
# ================================================================
# PIE CHART DISTRIBUTION VARIABLE
pie_data = df.loc[:, ['DPR412_classification']].value_counts().reset_index()
pie_data.columns = ["DPR412_classification", "count"]

# Prepare data for ECharts
chart_data = [{"name": row["DPR412_classification"], "value": int(row["count"])} for _, row in pie_data.iterrows()]
# ======

data = [{"values":0}]
# Remove Latitude and longitude
df_analysis = df.drop(columns=["latitude", "longitude"])

data_table = DashMantineReactTable(
        data=df_analysis.to_dict("records"),
        columns=[{"accessorKey": i, "header": i} for i in df_analysis.columns],
        mrtProps={
            "enableHiding": False,
            "enableColumnFilters": False,
            "initialState": {"density": "sm"},
            "mantineTableProps": {"fontSize": "sm"},
            "mantineTableHeadCellProps": {"style": {"fontWeight": 500}},
        },
        mantineProviderProps={
            "theme": {
                "colorScheme": "light",
            },
        },
    )

plots_univariate_distribution = dmc.Grid(
    children = [
        dmc.GridCol(
            children = [
                dmc.Select(
                    id="parameters", 
                    label = "Variable",
                    data = df.columns.to_list(),
                    value = "average_opaque_surface_transmittance",
                    clearable = False,
                    radius="md",
                )
            ], 
            span=4
        ), 
        dmc.GridCol(
            children = [
                dmc.AreaChart(
                    id='univariate_dist_areachart',
                    h=300,
                    data=data,
                    dataKey="values",
                    curveType="monotone",
                    tickLine="xy",
                    withXAxis=True,
                    withDots=False,
                    gridAxis=None,
                    withGradient=True, 
                    series=[
                        {"name": "values", "color": "pink.6"},
                    ],
                ),
            ],
            span=8
        )
    ],
    mt=10,
    mb=10
)

correlation_matrix = dash_echarts.DashECharts(
    id="correlation_heat_chart",
    style={
        "width": '100%',
        "height": "600px"
    }
)

# Infor classificazione DPR412
info_1 = dash_echarts.DashECharts(
        option={
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "name": "DPR412_classification",
                    "type": "pie",
                    "radius": ['40%', '70%'],
                    "data": chart_data,
                    "padAngle": 5,
                    "itemStyle": {
                        "borderRadius": 10
                    },
                    "label": {
                        "show": False,
                        "position": 'center'
                    },
                    "emphasis": {
                        "label": {
                        "show": True,
                        "fontSize": 40,
                        "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": False
                    },
                }
            ]
        },
        style={"height": "600px"}
    )

# initial_center = [45.02569105418987, 7.671092180850915]
# main_analysis = dmc.Stack(
#     children = [
#         data_table,
#         dmc.Text("Building map", size="xl", fw=800, mb=10, mt=5),
#         dmc.Grid(
#             children=[
#                 dmc.GridCol(
#                     dmc.Select(
#                         id="map_inputs", 
#                         label = "Variable",
#                         data = [
#                             {"label": "building typology", "value": "DPR412_classification"},
#                             {"label": "construction year", "value": "construction_year"},
#                         ],
#                         value = "construction_year",
#                         clearable = False,
#                         radius="md",
#                     ),
#                     span=3
#                 ),
#                 dmc.GridCol(
#                     id="map_bui",
#                     # children = [

#                     #         dl.Map(
#                     #             id="map_bui",
#                     #             center=initial_center,
#                     #             zoom=6,
#                     #             maxZoom=12,
#                     #             style={'width': '100%', 'height': '50vh'}
#                     #         ),
#                     # ],
#                     span=9
#                 )
#             ],
#             mt=10,
#             mb=10
#         ),
#         dmc.Text("Univariate Distribution of variable", size="xl", fw=800, mb=10, mt=5),
#         plots_univariate_distribution,
#         dmc.Text("Classification buildng type", size="xl", fw=800, mb=10, mt=5),
#         dmc.Grid(
#             children = [
#                 dmc.GridCol(
#                     children = dmc.ScrollArea(
#                         children = dmc.Table(
#                             data={
#                                 "caption": "Building category according to Italian DPR412",
#                                 "head": ["DPR 412", "code_EPC"],
#                                 "body": list_of_lists,
#                             }
#                         ),
#                         type="always",
#                         offsetScrollbars=True,
#                         h=500,
#                         mt=10
#                     ),
#                     span=6
#                 ),
#                 dmc.GridCol(
#                     children = info_1,
#                     span=6
#                 )
#             ]
#         ),
#         dmc.Text("Correlation Matrix", size="xl", fw=800, mb=10, mt=5),
#         correlation_matrix,
#     ]
# )



# ==================================================================================================================
#                                                LAYOUT OVERALL
# ==================================================================================================================

# layout_processing= html.Div(
#     children = [
#         dmc.AppShell(
#             [
#                 dcc.Loading([],
#                     custom_spinner = html.Span(className= "loader_spin_2"),
#                     style={'marginTop':'0px', 'marginBottom':'10px'},
#                     overlay_style={"visibility":"visible", "filter": "blur(2px)"},
#                 ),
#                 dmc.AppShellMain(
#                     children=[
#                         dmc.Container(
#                             size="xl",
#                             children = main_analysis
#                         )
#                         ],
#                     p=30
#                 ),
#             ]
#         ),
#     ],
#     style = {'marginBottom':'20px'}
# )


# def layout():
#     return layout_processing