# Library
import pandas as pd
from global_inputs import directory_
import folium 


def map_wurth(data, name_map):
    '''
    '''
    if len(data)!=0:
        # coord_city = [55,24]
        coord_city = [float(list(data['lat'])[0]), float(list(data['long'])[0])]
        map = folium.Map(tiles=None, location = coord_city,zoom_start = 3,)
        folium.raster_layers.TileLayer(tiles='cartodbpositron', location = coord_city,
                                    zoom_start = 3,
                                    name=name_map).add_to(map)
        
        lat_lst = data['lat']
        lng_lst = data['long']
        name_lst = data['buildingName']
        color_markers = data['color']     
        
        for lat, lng, name, color in zip(lat_lst, lng_lst, name_lst, color_markers):
            feature_group = folium.FeatureGroup("Buildings")
            feature_group.add_child(folium.Marker(location=[lat,lng],popup=name, icon=folium.Icon(color=color)))
            m = map.add_child(feature_group)
                    
        folium.LayerControl().add_to(m)
        
        # Generate Map
        # m.save(directory_ + f"/MATRYCS_ECM/frontend/maps/{name_map}.html")
        m.save(directory_ + f"/maps/{name_map}.html")
        
        nameMap =directory_ + f"/maps/{name_map}.html"
    else: 
        coord_city = [52.520008,13.404954] # Berlin
        map = folium.Map(tiles=None, location = coord_city,zoom_start = 3,)
        folium.raster_layers.TileLayer(tiles='cartodbpositron', location = coord_city,
                                    zoom_start = 3,
                                    name=name_map).add_to(map)
        # Generate Map
        map.save(directory_ + f"/maps/{name_map}.html")
        nameMap =directory_ + f"/maps/{name_map}.html"
    return nameMap



dataAll = pd.read_csv('TestWurth.csv', sep=",", decimal=".")
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

# ================================================================================



# ======================================================
from dash import  html, dcc, Input, Output, dash, Dash
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import dash_echarts
# import dash_leaflet as dl
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Location(id='url_app'),
        dmc.Container(
            children = [
                html.Div(
                    className = "row",
                    children = [
                        # ECHARTS 
                        html.Div(
                            className = "col-lg-12 col-md-12",
                            children = [
                                 dash_echarts.DashECharts(
                                    id = 'scatterBui',
                                    style={
                                        "width": '100%',
                                        "height": '350px',
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
                            ]
                        ),
                        html.Div(
                            className ="col-lg-12 col-md-12",    
                            children = [
                                dcc.Slider(
                                    dfMeanAll['Year'].min(),
                                    dfMeanAll['Year'].max(),
                                    step=None,
                                    id='year_slider',
                                    value=dfMeanAll['Year'].max(),
                                    marks={str(year): str(year) for year in dfMeanAll['Year'].unique()}
                                )
                            ],
                            style={'width': '49%', 'padding': '0px 20px 20px 20px'}
                        ),       
                    ]
                ),
                html.Div(
                    className = "row",
                    children = [
                        html.Div(
                            className ="col-lg-6 col-md-12",
                            children = [
                                dash_echarts.DashECharts(
                                    id = 'energyMonth',
                                    style={
                                        "width": '100%',
                                        "height": '350px',
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
                                        "height": '350px',
                                        },
                                )
                            ]
                            
                        )
                    ],
                ),
                html.Br(), 
                html.Div(
                    className = "col-lg-12 col-md-12",
                    children = [
                        html.Iframe(id = "mapBenchWurth",
                            width = '100%', height= '100vh',
                            style={"height": "20rem","width": "100%",'border':'None'},
                        )
                    ],
                ),
            ]
        ),
    ]
)

# grafico energy HVAc/m2 vs energy tot/m2
# possibilità di scegliere i negozi e paragonare i consumi 
# mappa geografica dei punti vendita con consumo media regionale tenendo conto del numero di edifici per regione (visualizzazioen a livello di nazione)
# cliccando sulla regione si apre la pagina della regione con i diversi punti vendita.

# fissare yaxes 

# Function time series
def create_time_series_1(dff, xaxes, yaxes, title, modeGraph):

    fig = px.scatter(dff, x=xaxes, y=yaxes)

    fig.update_traces(mode=modeGraph)

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')# if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig

# ================================================================================
#                       BUILDING IN MAP
# ================================================================================

import dash_echarts
@app.callback(
    Output("mapBenchWurth","srcDoc"),
    Input("scatterBui",'click_data'),
)
def visualize_building_in_homeMap(clickData):
    '''
    Visualize building in a map
    '''
    dfBuildings = dataAll.loc[:, ['Store identifier','latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    dfBuildings['color'] = 'red'
    dfBuildings.columns = ['buildingName', 'lat', 'long','color'] 
    
    if clickData is not None: 
        buiSelected = "BCFS"
        buiSelected= clickData['data'][2]
        # buiSelected = clickData['points'][0]['hovertext']
        dfBuildings.loc[dfBuildings['buildingName'] ==buiSelected, 'color'] = "green"
        
    # children = [dl.TileLayer(), leaflet_buildings(dfBuildings)]
    map = map_wurth(dfBuildings,'benchMap')
    return open(map,'r').read()
    

#  ================================================================================================
#                           MAIN SCATTER PLOT
#  ================================================================================================
def scatter_plot_main(source, title,  xAxesName, yAxesName):
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
    Output("scatterBui",'option'    ),
    Input('year_slider', 'value'),
    Input("scatterBui",'click_data')
)
def bui_main_scatter(year_value, clickGraph):
    '''
    '''
    
    dfY = dfMeanAll[dfMeanAll['Year'] == year_value]
    # Get yearly consumption 
    dfYEnergy = pd.DataFrame(dfY.groupby('building name')['Energy [kWh/m2]'].agg('sum')).reset_index()
    dfYEnergy['Ext Temperature'] = pd.DataFrame(dfY.groupby('building name')['External temperature (Celsius degree)'].agg('mean')).reset_index()['External temperature (Celsius degree)']
    source = []
    for i,element in dfYEnergy.iterrows():
        source.append([element['Ext Temperature'], element['Energy [kWh/m2]'], element['building name']])

    title = " Energy consumption vs Ext Temperature - Year"
    return scatter_plot_main(source, title, 'Ext.Temp.-°C', 'Energy [kWh/m2]')
   

#  ================================================================================================
#                           ENERGY PLOT MONTHLY 
#  ================================================================================================

def line_month_consumption(source, title, xAxesName, yAxesName,type):
    option = {
            'title': {
                'text':title,
                'nameGap': 0,
                'left':'center',
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
    Input("scatterBui",'click_data'),
    Input('year_slider', 'value'), 
    )
def update_timeseries(echartData, year):
    
    if echartData != None:
        buiName= echartData['data'][2]
    else:
        buiName = "BCFT"
    # Filter data according to the building selected
    dff = dfMeanAll[dfMeanAll['building name'] == buiName]
    dff = dff[dff['Year'] == year]
    
    # 
    xaxes = 'month'
    yaxes = 'Energy [kWh/m2]'
    dfToPlot = pd.DataFrame(dff.loc[:,[xaxes, yaxes]])
    title = f'Monthly energy profile - bui:{buiName}'
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
    Input("scatterBui",'click_data'),
    Input('year_slider', 'value'),
    )
def update_timeseries(echartData, year):#, feature):
    
    
    if echartData != None:
        buiName= echartData['data'][2]
    else:
        buiName = "BCFT"
    # Filter data according to the building selected
    dff = dfMeanAll[dfMeanAll['building name'] == buiName]
    # filtering by year
    dff = dff[dff['Year'] == year]
    # 
    xaxes = 'External temperature (Celsius degree)'
    yaxes = 'Energy [kWh/m2]'
    dfToPlot = pd.DataFrame(dff.loc[:,[xaxes, yaxes]])
    title = f'Monthly - Energy Consumption vs Ext.Temp - bui:{buiName}'
    source = []
    for i,element in dfToPlot.iterrows():
        source.append([element['External temperature (Celsius degree)'], 
                       element['Energy [kWh/m2]']])
    
    # ============================================
    echartsGraph = line_month_consumption(source,title,xaxes,yaxes,'scatter')

    return echartsGraph
 


#  ================================================================================================
#                           SCATTER BUI - ECHARTS
#  ================================================================================================
 

 
 
 
if __name__ == '__main__':
    app.run_server(debug=True)