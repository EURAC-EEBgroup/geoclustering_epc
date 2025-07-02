
# LIBRARIES 
from typing import List
import pandas as pd
import re


# ========================================================================
#                       CCLEANING FUNCTION
# ========================================================================

def Get_value_from_RDF_query(RDF_query):
    '''
    GET value from LEIF ttl file realted to energy consumption
    RDF_query: 
    
    Output: dataframe with information about type of consumption and value 
    '''
    df = pd.DataFrame(RDF_query, columns=RDF_query.vars)
    # Lista di tutti gli elementi
    Result = pd.DataFrame()
    for index_col in range(len(df.columns)):
        subdf = pd.DataFrame(df.iloc[:,index_col]).dropna()
        subdf = subdf.applymap(str)
        subdf = subdf.reset_index(drop=True)
        Result = pd.concat([Result, subdf], axis=1)

    return(Result)

def clean_result(data):
    """
    Clean data from unuseful element such as "urm:.."
    eg. from "urn:Gdyna#Air_Quality_Sensor_2123ccc6" -> "Air_Quality_Sensor_2123ccc6"
    INPUT:
    
    data: Series
    """
    data = pd.DataFrame(data)
    sensor_list  = []
    for index, row in data.iterrows():
        value = row[0]
        sensor_name = (re.findall(r'#(\w+)', value) or None,)[0]
        sensor_list.append(sensor_name[0])
      
    return sensor_list  


# ========================================================================
# import brickschema
# wurth = brickschema.Graph()
# wurth.load_file("wurth_shop.ttl")

# # Dictioanry with brickschema files
# Building_Dic = {}
# # Upload 
# Building_Dic['Bui_0242ac120002'] = wurth
# brick_graph = Building_Dic['Bui_0242ac120002']
# ========================================================================

#                   GET GENERAL INFO BUILDINGS
def get_all_bui_info(brick_graph):
    '''
    get all info from buis
    '''
    Query = """SELECT ?store ?lat ?long ?area ?location WHERE {
            ?store brick:latitude [ brick:value ?lat ] .     
            ?store brick:longitude [ brick:value ?long ] .  
            ?store brick:area [brick:value ?area ] .   
            ?store brick:isPartOf ?location .
        }
        """
        
    data_result = Get_value_from_RDF_query(brick_graph.query(Query))
    data_result.columns = ['store','lat','long','area','location']
    # df = pd.DataFrame(brick_graph.query(Query), columns=brick_graph.query(Query).vars)
    
    # Lista di tutti gli elementi
    right_bui_name  = []
    for index, row in data_result.iterrows():
        value = row[0]
        sensor_name = (re.findall(r'#(\w+)', value) or None,)[0]
        right_bui_name.append(sensor_name[0])    
    data_result['store'] = right_bui_name 
    
    # RIGHT LOCATION NAME
    right_location_name  = []
    for index, row in data_result.iterrows():
        value = row[4]
        elemnt = (re.findall(r'#(\w+)', value) or None,)[0]
        right_location_name.append(elemnt[0])    
    data_result['location'] = right_location_name    
    return data_result 