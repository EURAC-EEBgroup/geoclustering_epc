import requests
import pandas as pd
import numpy as np

import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign
from dash import html, Output, Input,dcc,Dash, State
# import dash_bootstrap_components as dbc
# import dash_admin_components as dac
import dash_mantine_components as dmc
import dash_echarts
from components.header import HeaderAdmin
from components.drawer import DrawerRight
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from dash_iconify import DashIconify


from global_inputs import buiGlobal,directory_


url_api = "http://127.0.0.1:8000"
def get_bui_info_wurth():
    '''
    GET ALL BUILDINGS
    '''

    url = f"{url_api}/api/v1/buildings/info"
    response = requests.request("GET", url)

    return pd.DataFrame(response.json())
buiGlobal = get_bui_info_wurth()

def leaflet_only_buildings():
    '''
    LEAFLET home page before cluster with only buildings
    '''

    # GET LIST OF ALL BUILDINGS
    df = buiGlobal.drop_duplicates(subset = ['store'])

    # FROM JSON TO DATAFRAME
    Buildings = pd.DataFrame(
        {
            "Building_name":list(df['store'].values),
            "lat": list(map(float, list(df['lat'].values))),
            "long": list(map(float, list(df['long'].values))),
            "Area":list(df['area'].values),
            "Location":list(df['location'].values),
        }
    )
    dicts = Buildings.to_dict(orient='records')

    # TOOLTIP
    for item in dicts:
        item["tooltip"] = (
            'Building_name:{} \n' \
            'Area:{} {}'
            ).format(item['Building_name'], item["Area"], "m2")


    geojson = dlx.dicts_to_geojson(dicts, lon="long")
    geobuf = dlx.geojson_to_geobuf(geojson)
    geojson = dl.GeoJSON(data=geobuf, id="geojson", format="geobuf",
                        zoomToBounds=True,  # when true, zooms to bounds when data changes
                        cluster=True,  # when true, data are clustered
                        zoomToBoundsOnClick=True,  # when true, zooms to bounds of feature (e.g. cluster) on click
    )

    return geojson


import folium


def map_wurth(data, name_map):
    '''
    '''
    coord_city = [float(list(data['lat'])[0]), float(list(data['long'])[0])]
    map = folium.Map(tiles=None, location = coord_city, zoom_start = 5)
    folium.raster_layers.TileLayer(tiles='cartodbpositron', location = coord_city,
                                zoom_start = 5,
                                name=name_map).add_to(map)

    if len(data)!=0:
        # coord_city = [55,24]

        lat_lst = data['lat']
        lng_lst = data['long']
        name_lst = data['buildingName']
        color_markers = data['color']

    else:

        lat_lst = 41.25925
        lng_lst = 14.46248
        name_lst = "BCFS"
        color_markers = data['color']

    for lat, lng, name, color in zip(lat_lst, lng_lst, name_lst, color_markers):
        feature_group = folium.FeatureGroup("Buildings")
        feature_group.add_child(folium.Marker(location=[lat,lng],popup=name, icon=folium.Icon(color=color)))
        m = map.add_child(feature_group)

    folium.LayerControl().add_to(m)

    # GENERATED MAP
    m.save(directory_ + f"/maps/{name_map}.html")
    nameMap =directory_ + f"/maps/{name_map}.html"

    return nameMap



dataAll = pd.read_csv('APP/TestWurth.csv')
#
dfMeanAll = pd.DataFrame()
for id in dataAll['Store identifier'].unique():
    # print(id)
    dataId = dataAll[dataAll['Store identifier']==id]
    for year in dataId['year'].unique():
        # print(year)
        data = dataId[dataId['year'] == year]
    # print(data)
        dfMean = pd.DataFrame(round(data.groupby('month')['External temperature (Celsius degree)'].mean(),2)).reset_index()
        dfMean['Energy [kWh/m2]'] = pd.DataFrame(round(data.groupby('month')['Global power (kW)'].sum()/data['Surface'].unique()[0],2)).reset_index()['Global power (kW)'].values
        dfMean['Year'] = year
        dfMean['building name'] = id
        dfMeanAll = pd.concat([dfMeanAll,dfMean])

# ====================================================================================================
#                           POLYNOMIAL TESTING
# ====================================================================================================

# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LinearRegression

# dff = dataAll[dataAll['Store identifier'] == 'BCFT']
# # filtering by year
# dff = dff[dff['year'] == 2021]
# # Filtering by month or all values
# dff = dff[dff['month'] == 5]
# dff = dff.dropna(subset=['External temperature (Celsius degree)'])
# X = pd.DataFrame({'x': list(dff['External temperature (Celsius degree)'].values)}).dropna()
# y = pd.DataFrame({'y': list(dff['HVAC power (kW)'].values)}).dropna()
# X_seq = np.linspace(min(X['x']), max(X['x']), 300).reshape(-1, 1)
# degree=1
# polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
# polyreg.fit(X,y)
# # flat_list = [item for sublist in polyreg.predict(X_seq) for item in sublist]
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(X,y)
# plt.plot(X_seq,polyreg.predict(X_seq),color="black")
# plt.title("Polynomial regression with degree "+str(degree))
# plt.show()

# # PREDICT
# dff1 = dataAll[dataAll['Store identifier'] == 'BCFT']
# # filtering by year
# dff1 = dff1[dff1['year'] == 2022]
# # Filtering by month or all values
# dff1 = dff1[dff1['month'] == 5]
# dff1 = dff1.dropna(subset=['External temperature (Celsius degree)'])
# X1 = pd.DataFrame({'x': list(dff1['External temperature (Celsius degree)'].values)}).dropna()
# y1 = pd.DataFrame({'y': list(dff1['HVAC power (kW)'].values)}).dropna()
# X1_seq = np.linspace(min(X1['x']), max(X1['x']), 300).reshape(-1, 1)

# plt.figure()
# plt.scatter(X1,y1)
# plt.plot(X1,polyreg.predict(X1),color="black")
# plt.title("Polynomial regression with degree "+str(degree))
# plt.show()


# ========================
dff = dataAll[dataAll['Store identifier'] == 'BCFT']
# filtering by year
dff = dff[dff['year'] == 2021]
# Filtering by month or all values
dff = dff[dff['month'] == 5]
dff = dff.dropna(subset=['External temperature (Celsius degree)'])

# PREDICT
dff1 = dataAll[dataAll['Store identifier'] == 'BCFT']
# filtering by year
dff1 = dff1[dff1['year'] == 2022]
# Filtering by month or all values
dff1 = dff1[dff1['month'] == 5]
dff1 = dff1.dropna(subset=['External temperature (Celsius degree)'])

X_train = dff['External temperature (Celsius degree)'].values
Y_train = dff['HVAC power (kW)'].values
X_val = dff1['External temperature (Celsius degree)'].values
Y_val = dff1['HVAC power (kW)'].values
degree=2
model=make_pipeline(PolynomialFeatures(degree),LinearRegression())
model.fit(X_train.reshape(-1,1), Y_train)
Y_pred = model.predict(X_val.reshape(-1,1))





# import pandas as pd
# import numpy as np
# pd.options.plotting.backend = "plotly"


# df = pd.DataFrame(dict(
#     a=Y_val,
#     b=Y_pred
# ))
# fig = df.plot()
# fig.show()







# ====================================================================================================


app = Dash(__name__)

# ====================================================================================================





graphPage1 = html.Div(
    className = "card",
    style = {
        'backgroundColor':'#f6f6f6'
    },
    children = [
        html.Div(
            className = "card-body",
            children = [
                html.Div(
                    className ="row",
                    children = [
                        dash_echarts.DashECharts(
                            id = 'scatterBui',
                            style={
                                "width": '100%',
                                "height": '300px',
                                },
                            funs={
                                "BuildingName":
                                '''
                                function(params){
                                    return "Building:" + "<br>"  + "<b>"+params[0].data[2]+"</b>"
                                }
                                '''
                            },
                            fun_values=['BuildingName'],
                        ),
                        dmc.Text(
                            "Period - Year:",
                            color = "#6c757d",
                            style={
                                'fontSize': 20,
                                'fontWeight':600
                                }
                        ),
                        html.Br(),
                        dcc.Slider(
                            dfMeanAll['Year'].min(),
                            dfMeanAll['Year'].max(),
                            step=None,
                            id='year_slider',
                            value=dfMeanAll['Year'].max(),
                            marks={str(year): str(year) for year in dfMeanAll['Year'].unique()}
                        ),
                        html.Hr(),
                        html.Div(
                            className ="col-lg-6 col-md-12",
                            children = [
                                dash_echarts.DashECharts(
                                    id = 'energyMonth',
                                    style={
                                        "width": '100%',
                                        "height": '300px',
                                        },
                                )
                            ]
                        ),
                        html.Div(
                            className ="col-lg-6 col-md-12",
                            children = [
                                dash_echarts.DashECharts(
                                    id = 'EnergyFigure_ExtTemp_month',
                                    style={
                                        "width": '100%',
                                        "height": '300px',
                                        },
                                )
                            ]
                        ),
                        html.Div(
                            className ="col-lg-12 col-md-12",
                            children = [
                                html.Div(
                                    className = "card",
                                    children = [
                                        html.Div(
                                            className = "card-body",
                                            children = [
                                                dash_echarts.DashECharts(
                                                    id = 'HeatMap',
                                                    style={
                                                        "width": '100%',
                                                        "height": '300px',
                                                        },
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


RegressionPage = html.Div(
        className = "row",
        children = [
            html.Div(
                className = "col-lg-2 col-md-12",
                children = [
                    dmc.Select(
                        id = "monthregression",
                        label= "Time Selection",
                        data = [
                            {'label': 'Jan', 'value': 1},
                            {'label': 'Feb', 'value': 2},
                            {'label': 'Mar', 'value': 3},
                            {'label': 'Apr', 'value': 4},
                            {'label': 'May', 'value': 5},
                            {'label': 'Jun', 'value': 6},
                            {'label': 'Jul', 'value': 7},
                            {'label': 'Aug', 'value': 8},
                            {'label': 'Sep', 'value': 9},
                            {'label': 'Oct', 'value': 10},
                            {'label': 'Nov', 'value': 11},
                            {'label': 'Dec', 'value': 12},
                            {'label': 'All', 'value': 'all'},
                            ],
                        value = 1
                    ),
                    html.Br(),
                    dmc.Select(
                        id="regressionType",
                        label = "Regression type",
                        data = ['linear','polynomial','exponential', 'logarithmic'],
                        value= "linear"
                    ),
                    html.Br(),
                    dmc.NumberInput(
                        id="modelOrder",
                        label="Model Order",
                        value=1,
                        min=0,
                        step=1,
                        disabled=True
                    ),
                    html.Hr(), 
                    dmc.Switch(
                        id="benchRegression",
                        label = "Compare periods",
                        thumbIcon=DashIconify(
                            icon="tabler:walk", width=16, color=dmc.theme.DEFAULT_COLORS["teal"][5]
                        ),
                        size="md",
                        color="teal",
                        checked=True,
                    ),
                    dmc.Select(
                        id="yearBench2",
                        label = "Year",
                        description = "Year for the secondo period to benchmark",
                        data = [
                            {'label':'2021', 'value':2021},
                            {'label':'2022', 'value':2022}
                            ],
                        value= 2022,
                        disabled =True
                    ),
                ]
            ),
            html.Div(
                className="col-lg-9 col-md-12",
                children = [
                    dash_echarts.DashECharts(
                        id = 'regressionPlot',
                        style={
                            "width": '100%',
                            "height": '300px',
                            }
                    ),
                    html.Div(
                        children = [
                            dash_echarts.DashECharts(
                                id = 'regressionPlotBench',
                                style={
                                    "width": '100%',
                                    "height": '300px',
                                    "display": 'block'
                                }
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id="Test"
            )
        ]
    )


# Analysis1= html.Div(
#     className = "card",
#     children = [
#         html.Div(
#             className="card-body",
#             children=[
#                 graphPage1
#             ]
#         )
#     ]
# )

# Accordion = dmc.Accordion(
#     disableChevronRotation=True,
#     variant = 'filled',
#     children=[
#         dmc.AccordionItem(
#             [
#                 dmc.AccordionControl(
#                     "Data Visualization",
#                     icon=DashIconify(
#                         icon="tabler:user",
#                         color=dmc.theme.DEFAULT_COLORS["blue"][6],
#                         width=20,
#                     ),
#                 ),
#                 dmc.AccordionPanel(
#                     graphPage1
#                 )    
#             ],
#             value="info",
#         ),
#         # dmc.AccordionItem(
#         #     [
#         #         dmc.AccordionControl(
#         #             "Data Model",
#         #             icon=DashIconify(
#         #                 icon="tabler:map-pin",
#         #                 color=dmc.theme.DEFAULT_COLORS["red"][6],
#         #                 width=20,
#         #             ),
#         #         ),
#         #         dmc.AccordionPanel(
#         #             RegressionPage
#         #             ),
#         #     ],
#         #     value="addr",
#         # ),
#     ],
#     value = ["info","addr"]
# )


app.layout = html.Div(
    children = [
        html.Div(
            id = "page-content-home",
            children = [
                HeaderAdmin,
                DrawerRight,
                html.Div(
                    className = "row",
                    children = [
                        dmc.Text(
                            id = "buiNameSelected",
                            color = "black",
                            style={
                                'fontSize': 30,
                                'fontWeight':800,
                                'marginTop':'75px'}
                        ),
                        html.Hr(),
                        html.Div(
                            className = "col-lg-4 col-md-12",
                            children = [
                                html.Div(
                                    className ="card",
                                    children = [
                                        html.Div(
                                            className="card-body",
                                            children = [
                                                dl.Map([dl.TileLayer(), leaflet_only_buildings()],
                                                center=[11.464212786854729, 43.46729850674922],
                                                # animate = True,
                                                id="map_1", style={ 'height': '100%', 'margin': "auto","position": "relative","display": "block", 'zIndex':'50', 
                                                                'marginTop':'5px'} )
                                            ]    
                                        )
                                    ]
                                )
                                # dmc.Stack(
                                    # children = [
                                        #  html.Iframe(id = "mapBenchWurth",
                                        #     width = '100%', height= '100vh',
                                        #     # style={"height": "20rem","width": "100%",'border':'None'},
                                        #     style={ 'height': '65vh', 'margin': "auto","position": "relative","display": "block", 'zIndex':'50'}
                                        # ),
                                        
                                    # ]
                                # )
                            ],
                            style = {'display':'grid'}
                        ),
                        html.Div(
                            className= "col-lg-8 col-md-12",
                            children = graphPage1,
                            style = {'display':'grid'}
                            
                            # children = [
                            #     html.Div(
                            #         className = "row",
                            #         id = "graphPageSelected",
                            #         children =  Analysis1
                            #     )
                            # ],
                        ),
                    ]
                ),
                html.Hr(),
                html.Div(
                    className = "card",
                    children = [
                        html.Div(
                            className="card-body",    
                            children = [
                                RegressionPage
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

@app.callback(
    Output("Test","children"),
    Input("regressionPlot","click_data")
)
def visualize_inputs(option):
    print(option)


# ========================================================================================
#                   PAGES ACCORDIN TO THE SLIDER SELECTION
# ========================================================================================

# @app.callback(
#     Output("graphPageSelected",'children'),
#     Input("sliderPages",'value')
# )
# def pages(slideVal):
#     '''

#     '''
#     if slideVal ==1:
#         page = ""
#     else:
#         page = graphPage1

#     return page



# ============================================
#                   OPEN DRAWER RIGHT
# ============================================

@app.callback(
    Output('drawerRight', 'opened'),
    Input('rightDrawer', 'n_clicks'),
    State('drawerRight', 'opened'),
    prevent_initial_call=True,
)
def drawer_demo(n_clicks,opened):
    return not opened


# ============================================
#                   NAME OF THE SELECTED BUILDING
# ============================================


# @app.callback(
#     Output("buiNameSelected", 'children'),
#     # Input("geojson", "click_feature")
#     Input("scatterBui",'click_data'),
# )
# def get_bui_name(buiFeature):
#     '''

#     '''
#     # if buiFeature:
#     #     if buiFeature['properties']['cluster'] == True:
#     #         buiNameTitle = ""
#     #     else:
#     #         buiName= buiFeature['properties']['Building_name']
#     #         location = buiFeature['properties']['Location']
#     #         buiNameTitle = buiName + "-" + location
#     # else:
#     #     buiName = "BCGK"
#     #     buiNameTitle= buiName + '-Osimo'
#     if buiFeature is not None:
#         buiSelected= buiFeature['data'][2]
#         # location = buiFeature['properties']['Location']
#         buiNameTitle = buiSelected
#     else:
#         buiNameTitle=  'BCFS'

#     return f'Building Selected: {buiNameTitle}'



# ============================================
#                   ACTIVATE MODEL ORDER
# ============================================

@app.callback(
    Output('modelOrder','disabled'),
    Input('regressionType','value')
)
def activate_orderModel(type):
    if type != 'linear':
        return False
    return True



@app.callback(
    Output('yearBench2','disabled'),
    # Output('regressionPlotBench','style'),
    Input('benchRegression','checked')
)
def activate_orderModel(benchRegression):
    if benchRegression == True:
        return False#, {'display':'block'}
    return True#, {'display':'none'}



# ================================================================================
#                       BUILDING IN MAP
# ================================================================================

# import dash_echarts
# @app.callback(
#     Output("mapBenchWurth","srcDoc"),
#     Input("scatterBui",'click_data'),
# )
# def visualize_building_in_homeMap(clickData):
#     '''
#     Visualize building in a map
#     '''
#     # GET BUILDINGS FROM RDF Database
#     dfBuildings = buiGlobal.drop_duplicates(subset = ['store'])

#     # dfBuildings = dataAll.loc[:, ['Store identifier','latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
#     # dfBuildings['color'] = 'red'
#     # dfBuildings.columns = ['buildingName', 'lat', 'long','color']
#     dfBuildings['color'] = 'red'
#     dfBuildings.columns = ['buildingName', 'lat', 'long','area','location', 'color']

#     if clickData is not None:
#         buiSelected= clickData['data'][2]
#         # buiSelected = clickData['points'][0]['hovertext']
#         dfBuildings.loc[dfBuildings['buildingName'] ==buiSelected, 'color'] = "green"
#     else:
#         dfBuildings.loc[dfBuildings['buildingName'] =="BCFS", 'color'] = "green"
#         buiSelected = "BCFS"

#     # children = [dl.TileLayer(), leaflet_buildings(dfBuildings)]
#     # return children
#     map = map_wurth(dfBuildings,'benchMap')
#     return open(map,'r').read()


#  ================================================================================================
#                           MAIN SCATTER PLOT
#  ================================================================================================
def scatter_plot_main(source, title,  xAxesName, yAxesName, source2):
    option = {
        'title': {
            'text':title
            },
        'dataset': [
            {
                'source': source
            },
        ],
        'xAxis': {
            'name':xAxesName,
            'splitLine': {
                'lineStyle': {
                    'type': 'dashed'
                }
            }
        },
        'yAxis': {
            'name': yAxesName,
            'splitLine': {
                'lineStyle': {
                    'type': 'dashed'
                }
            }
        },
        'toolbox': {
            'show': True,
            'feature': {
            'dataZoom': {
                'yAxisIndex': 'none'
            },
            'dataView': { 'readOnly': False },
            'magicType': { 'type': ['line', 'bar'] },
            'saveAsImage': {}
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'formatter': 'BuildingName'
            },
        'series': [
            {
                'type': 'effectScatter',
                'symbolSize': 20,
                'data': source2
                },
            {
            'name': 'Test',
            'type': 'scatter'
            },
            {
            'name': 'line',
            'type': 'line',
            'datasetIndex': 1,
            'symbolSize': 0.1,
            'symbol': 'circle',
            'label': { 'show': True, 'fontSize': 16 },
            'labelLayout': { 'dx': -20 },
            'encode': { 'label': 2, 'tooltip': 1 }
            }
        ]
    }
    return option

@app.callback(
    Output("scatterBui",'option'),
    Output("buiNameSelected", 'children'),
    Input('year_slider', 'value'),
    Input("geojson", "click_feature")
)
def bui_main_scatter(year_value, buiFeature):
    '''
    '''
    if buiFeature:
        if buiFeature['properties']['cluster'] == True:
            buiNameTitle = ""
        else:
            buiName= buiFeature['properties']['Building_name']
            location = buiFeature['properties']['Location']
            buiNameTitle = buiName + "-" + location
    else:
        buiName = "BCG7"
        buiNameTitle= buiName + '- Nola'


    dfY = dfMeanAll[dfMeanAll['Year'] == year_value]
    # Get yearly consumption
    dfYEnergy = pd.DataFrame(dfY.groupby('building name')['Energy [kWh/m2]'].agg('sum')).reset_index()
    dfYEnergy['Ext Temperature'] = pd.DataFrame(dfY.groupby('building name')['External temperature (Celsius degree)'].agg('mean')).reset_index()['External temperature (Celsius degree)']

    # Remove selected building
    dfYEnergyNotSelected = dfYEnergy[dfYEnergy['building name'] != buiName]

    # List of builings not selected
    source = []
    for i,element in dfYEnergyNotSelected.iterrows():
        source.append([element['Ext Temperature'], element['Energy [kWh/m2]'], element['building name']])

    # list for buildings selected
    dfYEnergyBuiSel = dfYEnergy[dfYEnergy['building name'] == buiName]
    sourceBuiSel = []
    for i,element in dfYEnergyBuiSel.iterrows():
        sourceBuiSel.append([element['Ext Temperature'], element['Energy [kWh/m2]'], element['building name']])
    # print(dfYEnergyBuiSel)
    # ============================================================
    title = " Energy consumption vs Ext Temperature - Year"

    return scatter_plot_main(source, title, 'Ext.Temp.-°C', 'Energy [kWh/m2]', sourceBuiSel), buiNameTitle


#  ================================================================================================
#                           ENERGY PLOT MONTHLY
#  ================================================================================================

def line_month_consumption(source, title, xAxesName, yAxesName,type):
    option = {
            'title': {
                'text':title,
                'nameGap': 0,
                # 'left':'center',
                # 'nameTextStyle': {
                #     'align': 'right',
                #     'verticalAlign': 'top',
                #     'padding': [0, 50, 0, 0],
                # }
                },
            'dataset': [
                {
                    'source': source
                },
            ],
            'xAxis': {
                'name':xAxesName,
                'nameLocation': 'end',
                'nameGap': 0,
                'nameTextStyle': {
                    'align': 'right',
                    'verticalAlign': 'top',
                    'padding': [30, 0, 0, 0],
                }
            },
            'yAxis': {
                'name': yAxesName,
            },
            'toolbox': {
                'show': True,
                'feature': {
                'dataZoom': {
                    'yAxisIndex': 'none'
                },
                'dataView': { 'readOnly': False },
                'magicType': { 'type': ['line', 'bar'] },
                'saveAsImage': {}
                }
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                'type': 'cross'
                }
            },
            'series': [
                {
                'name': 'Energy Month',
                'type': type,
                'smooth': True
                },
                {
                # 'name': 'line',
                # 'type': 'line',
                # 'datasetIndex': 1,
                # 'symbolSize': 0.1,
                # 'symbol': 'circle',
                # 'label': { 'show': True, 'fontSize': 16 },
                # 'labelLayout': { 'dx': -20 },
                # 'encode': { 'label': 2, 'tooltip': 1 }
                }
            ]
        }
    return option

@app.callback(
    Output('energyMonth','option'),
    Input("geojson", "click_feature"),
    # Input("scatterBui",'click_data'),
    Input('year_slider', 'value'),
    )
def update_timeseries(buiFeature, year):

    if buiFeature:
        if buiFeature['properties']['cluster'] == True:
            buiName = ""
        else:
            buiName= buiFeature['properties']['Building_name']
    else:
        buiName = "BCG7"
    # print(buiName) 
    # if echartData != None:
    #     buiName= echartData['data'][2]
    # else:
    #     buiName = "BCFT"
    # Filter data according to the building selected
    dff = dfMeanAll[dfMeanAll['building name'] == buiName]
    dff = dff[dff['Year'] == year]

    #
    xaxes = 'month'
    yaxes = 'Energy [kWh/m2]'
    dfToPlot = pd.DataFrame(dff.loc[:,[xaxes, yaxes]])
    title = f'Monthly energy profile'
    source = []
    for i,element in dfToPlot.iterrows():
        source.append([element['month'], element['Energy [kWh/m2]']])

    # ============================================
    echartsGraph = line_month_consumption(source,title,xaxes,yaxes,'line')

    return echartsGraph


#  ================================================================================================
#                           ENERGY EXT TEMP MONTHLY
#  ================================================================================================
@app.callback(
    Output('EnergyFigure_ExtTemp_month', 'option'),
    Input("geojson", "click_feature"),
    # Input("scatterBui",'click_data'),
    Input('year_slider', 'value'),
    )
def update_timeseries(buiFeature, year):#, feature):

    if buiFeature:
        if buiFeature['properties']['cluster'] == True:
            buiName = ""
        else:
            buiName= buiFeature['properties']['Building_name']
    else:
        buiName = "BCG7"
    
    # if echartData != None:
    #     buiName= echartData['data'][2]
    # else:
    #     buiName = "BCFT"
    # Filter data according to the building selected
    dff = dfMeanAll[dfMeanAll['building name'] == buiName]
    # filtering by year
    dff = dff[dff['Year'] == year]
    #
    xaxes = 'External temperature (Celsius degree)'
    yaxes = 'Energy [kWh/m2]'
    dfToPlot = pd.DataFrame(dff.loc[:,[xaxes, yaxes]])
    title = f'Monthly - Energy Consumption vs Ext.Temp'
    source = []
    for i,element in dfToPlot.iterrows():
        source.append([element['External temperature (Celsius degree)'],
                       element['Energy [kWh/m2]']])

    # ============================================
    echartsGraph = line_month_consumption(source,title,xaxes,yaxes,'scatter')

    return echartsGraph




#  ================================================================================================
#                           ENERGY EXT TEMP MONTHLY
#  ================================================================================================

def Barchart_comparison(title, subTitle, xDAta, nameSeries1, dataSeries1,nameSeries2, dataSeries2):
    '''
    xDAta; list
    '''
    option = {
        'title': {
            "text": title,
            "subtext": subTitle
        },
        "tooltip": {
            "trigger": 'axis'
        },
        "toolbox": {
            "show": True,
            "feature": {
            "dataView": { "show": True, "readOnly": False },
            "magicType": { "show": True, "type": ['line', 'bar'] },
            "restore": { "show": True },
            "saveAsImage": { "show": True }
            }
        },
        "calculable": True,
        "xAxis": [
            {
            "type": 'category',
            'data': xDAta
            }
        ],
        "yAxis": [
            {
            "type": 'value'
            }
        ],
        "series": [
            {
            "name": nameSeries1,
            "type": 'bar',
            "data": dataSeries1
            },
            {
            "name": nameSeries2,
            "type": 'bar',
            "data": dataSeries2,
            }
        ]
    }
    return option 

def RegressionChart(source, title, xName,yName, namePoint, regressionType, order):
    option = {
    'dataset': [
        {
        'source': source
        },
        {
        'transform': {
            'type': 'ecStat:regression',
            # // 'linear' by default.
            'config': {
                'method': regressionType,
                'order':order,
                'formulaOn': 'end'}
        }
        }
    ],
    'toolbox': {
        'show': True,
        'feature': {
        'dataZoom': {
            'yAxisIndex': 'none'
        },
        'dataView': { 'readOnly': False },
        'magicType': { 'type': ['line', 'bar'] },
        'saveAsImage': {}
        }
    },
    'title': {
        "text": title,
        "subtext": "regression building model"
    },
    'legend': {
        'bottom': 5
    },
    'tooltip': {
        'trigger': 'axis',
        'axisPointer': {
        'type': 'cross'
        }
    },
    'xAxis': {
        'name':xName,
        'splitLine': {
            'lineStyle': {
                'type': 'dashed'
            }
        },
        'nameTextStyle': {
            'align': 'right',
            'verticalAlign': 'top',
            'padding': [30, 0, 0, 0],
        }
    },
    'yAxis': {
        'name': yName,
        'splitLine': {
            'lineStyle': {
                'type': 'dashed'
            }
        },
        "nameTextStyle": {
            "align": 'right',
            "verticalAlign": 'top',
            "padding": [10, 0, 0, 10],
        }   
    },
    'series': [
        {
            'name': namePoint,
            'type': 'scatter'
        },
        {
            'name': 'regression',
            'type': 'line',
            'datasetIndex': 1,
            'symbolSize': 0.1,
            'symbol': 'circle',
            'label': { 'show': True, 'fontSize': 16 },
            'labelLayout': { 'dx': -20 },
            'encode': { 'label': 2, 'tooltip': 1 }
        }
    ]
    }
    return option

@app.callback(
    Output("regressionPlot", "option"),
    Output("regressionPlotBench", "option"),
    Input("geojson", "click_feature"),
    # Input("scatterBui",'click_data'),
    Input('year_slider', 'value'),
    Input('monthregression','value'),
    Input('regressionType','value'),
    Input('modelOrder','value'),
    Input('yearBench2','value'),
    Input('benchRegression','checked')    
)
def regressioPlot(buiFeature, year, month, regType, orderModel, year2, benchCheked):

    
    if buiFeature:
        if buiFeature['properties']['cluster'] == True:
            buiName = ""
        else:
            buiName= buiFeature['properties']['Building_name']
    else:
        buiName = "BCG7"
    # if echartData != None:
    #     buiName= echartData['data'][2]
    # else:
    #     buiName = "BCFT"
    # Filter data according to the building selected
    dff = dataAll[dataAll['Store identifier'] == buiName]
    dfStore= dff
    # filtering by year
    dff = dff[dff['year'] == year]
    # print(dff)

    # Filtering by month or all values
    if month !='all':
        dff = dff[dff['month'] == month]

    xaxes = 'External temperature (Celsius degree)'
    yaxes = 'HVAC power (kW)'
    dfToPlot = pd.DataFrame(dff.loc[:,[xaxes, yaxes]])
    print(dfToPlot)
    title = "Hourly Energy Consumption vs Ext Temperature"

    source = []
    for i,element in dfToPlot.iterrows():
        source.append([element['External temperature (Celsius degree)'],
                       element['HVAC power (kW)']])

    yaxesPlot = "Energy [kWh]"
    
    if benchCheked==True:
    # =============================================
    #              SIMULATE RGERESSION MODEL IN POST
    # =============================================
        # REMOVE NA VALUES
        dff = dff.dropna(subset=['External temperature (Celsius degree)'])

        # PREDICTED PERIOD
        # filtering by year
        dff1 = dfStore[dfStore['year'] == year2]
        # Filtering by month or all values
        if month !='all':
            dff1 = dff1[dff1['month'] == month]
        # REMOVE NA VALUES
        dff1 = dff1.dropna(subset=['External temperature (Celsius degree)'])

        # INPUT MODEL 
        X_train = dff['External temperature (Celsius degree)'].values
        Y_train = dff['HVAC power (kW)'].values
        X_val = dff1['External temperature (Celsius degree)'].values
        Y_val = dff1['HVAC power (kW)'].values
        degree=orderModel
        
        # CREATE MODEL
        model=make_pipeline(PolynomialFeatures(degree),LinearRegression())
        model.fit(X_train.reshape(-1,1), Y_train)
        
        # PREDICT DATA ON NEW PERIOD
        Y_pred = model.predict(X_val.reshape(-1,1))
        

        # import pandas as pd
        # import numpy as np
        # pd.options.plotting.backend = "plotly"


        # df = pd.DataFrame(dict(
        #     a=Y_val,
        #     b=Y_pred
        # ))
        # fig = df.plot()
        # fig.show()

        # # ============================================
        
        # ============================================
        x1 = list(range(0, len(Y_val),1))

        
        echartsComparison = Barchart_comparison('Actual and Predicted values','Prediction of energy consumption based on a building model', x1, "actual value",Y_val.tolist(),"predicted value",Y_pred.tolist())
    else:
        echartsComparison = Barchart_comparison('Actual and Predicted values','Prediction of energy consumption based on a building model', [], "actual value",[],"predicted value",[])
        # echartsComparison = option_white
        
    echartsGraph = RegressionChart(source,title,xaxes,yaxesPlot, "", regType, orderModel)
    return echartsGraph, echartsComparison



# ========================================================================================
#                               HEAT MAP -
# ========================================================================================
def heatmap(data1, days, hours):
    option = {
    "tooltip": {
        "position": 'top',
        'trigger': 'axis',
        'axisPointer': {
            'type': 'cross'
        }
    },
    "grid": {
        "height": '50%',
        "top": '10%'
    },
    'toolbox': {
        'show': True,
        'feature': {
        'dataZoom': {
            'yAxisIndex': 'none'
        },
        'dataView': { 'readOnly': False },
        'saveAsImage': {}
        }
    },
    "xAxis": {
        "type": 'category',
        "data": days,
        "splitArea": {
            "show": True
        },
        'name':'Days'
        # "min":0,
        # "max":30
    },
    "yAxis": {
        "type": 'category',
        "data": hours,
        "splitArea": {
            "show": True
        },
        'name':'Hour'
        # "min":1,
        # "max":24
    },
    "visualMap": {
        "min": 0,
        "max": 10,
        "calculable": True,
        "orient": 'horizontal',
        "left": 'center',
        "bottom": '15%'
    },
    "series": [
        {
        "name": 'Punch Card',
        "type": 'heatmap',
        "data": data1,
        # "label": {
        #     "show": True
        # },
        "emphasis": {
            "itemStyle": {
            "shadowBlur": 10,
            "shadowColor": 'rgba(0, 0, 0, 0.5)'
            }
        }
        }
    ]
    }
    return option

@app.callback(
    Output('HeatMap','option'),
    Input('Test','children')
)
def test(gg):
    buiName = "BCG7"
    dff = dataAll[dataAll['Store identifier'] == buiName]
    dff = dff[dff['month'] == 11]
    dff['Hour'] = pd.DatetimeIndex(dff['Date and time']).hour
    
    dataToPlot = []
    for i, data in dff.iterrows():
        dataToPlot.append([data['day'],data['Hour'],data['HVAC power (kW)']])
    
    y= [str(x) for x in list(pd.DatetimeIndex(dff['Date and time']).hour)]
    x= [str(x) for x in list(dff['day'])]
    
    return heatmap(dataToPlot, list(dict.fromkeys(x)), list(dict.fromkeys(y)))







if __name__ == '__main__':
    app.run_server(debug=True, port=8082, dev_tools_ui=False)

