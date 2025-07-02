from fastapi import FastAPI, Response, Depends
from typing import Union, List
from pydantic import BaseModel
import pandas as pd
import brickschema
from utils import get_all_bui_info
import utils as Utl

# ====================================================================================
#                   UPLOAD WURTH DATA
# ====================================================================================

wurth = brickschema.Graph()
wurth.load_file("wurth_shop.ttl")

# Dictioanry with brickschema files for each building
buildings_ = {}
# Upload ttl building file
buildings_['Bui_0242ac120002'] = wurth

# ====================================================================================
#                   UPLOAD WURTH DATA
# ====================================================================================

description_API = """
API to retrieve data from wurth files
"""

tags_metadata = [
    {
        "name": "Building information",
        "description": "retrieve buildings list according to specific features"
    },
]

# ====================================================================================
#                   INIZIALIZE FAST API 
# ====================================================================================

app = FastAPI(
    title = "WURTH - MODERATE ",
    description = description_API,
    version = "0.0.1",
    # terms_of_service = "http://example.com/terms/",
    contact = {
        "name" : 'Daniele Antonucci',
        "email" : "daniele.antonucci@eurac.edu",
    },
    license_info = {
        "name" : "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    },
    openapi_tags=tags_metadata
)

# ===============================================================================
#                               ENDPOINTS
# ===============================================================================
'''
GETs building list
'''
@app.get("/api/v1/buildings/info", tags= ['Building Information'])
async def get_building_list():
    '''
    Get list of buildings with the following information: 
    - Building Name
    - Location **
    - Latitude
    - Longitude
    - Store Area
    - Number of Zone **
    '''
    data_result = pd.DataFrame()
    for bui in buildings_:
        print(bui)
        data_result = pd.concat([data_result,Utl.get_all_bui_info(buildings_[bui])])
    return Response(data_result.to_json(orient="records"), media_type="application/json")

