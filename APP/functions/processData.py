import requests
import json 
import pandas as pd
from global_inputs import url_api


def get_bui_info_wurth():
    '''
    GET ALL BUILDINGS
    '''
    
    url = f"{url_api}/api/v1/buildings/info"
    response = requests.request("GET", url)
    
    return pd.DataFrame(response.json())