
#                              LIBRARY
# ============================================================================================ 

import json
import requests
import pandas as pd
import re

import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign
from dash import html
import dash_bootstrap_components as dbc
import dash_admin_components as dac
from global_inputs import buiGlobal


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
