import requests
import pandas as pd
import os
directory_ = os.getcwd()

# API BACKEND
url_api = "http://127.0.0.1:8000" 

# UPLOAD ALL BUILDINGS 
def get_bui_info_wurth():
    '''
    GET ALL BUILDINGS
    '''
    
    url = f"{url_api}/api/v1/buildings/info"
    response = requests.request("GET", url)
    
    return pd.DataFrame(response.json())
buiGlobal = get_bui_info_wurth()