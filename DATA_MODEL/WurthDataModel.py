# Library
import pandas as pd

BCFS = "../WURTH_/DATA/stores_data/BCFS.csv"
fileName = "BCFS"
data = pd.read_csv(BCFS,  sep=";", decimal=",")


# # GENERAL INFO OF DATASET
generalInfo = pd.read_csv("../WURTH_/DATA/stores.csv", sep=";", decimal=",")

# ========================================================================================
#                           DATA MODEL ACCORDING to FIWARE
# ========================================================================================
'''
Copyright...


'''

# BRICK SCHEMA 
import brickschema
from brickschema.namespaces import A, OWL, BRICK, UNIT, RDF, RDFS, XSD
from rdflib import Namespace, Literal
import pandas as pd

dataAll = pd.read_csv('../WURTH_/DATA_MODEL/TestWurth.csv')

# NOTE
# Called the name of the building in which the store is loacted as: 
# "bui_Store Name_storeIdentifier". E.g. bui_Nola_BCFS"  
# StoreName  == name of the city 


# our entities will live in this namespace
BLDG = Namespace("urn:example#")
STORE = Namespace("store#")


# create a graph for our model
g = brickschema.Graph()
# Define  a building
g.bind("bldg", BLDG)
g.bind("brick", BRICK)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("store", STORE)

def brick_data_model(storeIdentifier, storeName, storeArea, latStore, longStore, numberHVACZone):
    '''
    Create brick schema model from WURTH dataset 
    The building information provided by WURTH are:
    - Building Name ('Store identifier')
    - Store Location ('store name') = city
    - Building Location ('City)
    - Latitude
    - Longitude
    - Number of areas/rooms of the building
    - Indoor temperature
    - Outdoor Temperature
    - HVAC power consumption 
    - HVAC power consumption of the building - GLOBAL
    - Building Surface
    '''
    
    # =================================================
    #           SPACE DEFINITION
    # =================================================
    #                BUILDING ENTITY 
    '''define a general building in which the store is located '''
    # REPLACE WHITESPACE 
    storeName = storeName.replace(" ", "")
    storeIdentifier =storeIdentifier.replace(" ", "")
    buiOfStore = "bui"+storeName+storeIdentifier
    
    # Define latitude and longitude
    lat = [(BRICK.value, Literal(latStore, datatype=XSD.double))]
    long = [(BRICK.value, Literal(longStore, datatype=XSD.double))]
    
    # define a Building in a site
    g.add((BLDG[storeName], A, BRICK.Site))
    g.add((BLDG[buiOfStore], A, BRICK.Building))
    g.add((BLDG[storeName], BRICK.hasPart, BLDG[buiOfStore]))


    #                   STORE ENTITY
    '''Defining the 'store' namespace to identify the entities 
    that are part of the store graph vs the entities/concepts that are part of Brick 
    '''
    g.add((BRICK["Store"], RDFS.subClassOf, BRICK.Space))
    g.add((STORE[storeIdentifier], A, BRICK["Store"]))
    g.add((BLDG[buiOfStore], BRICK.hasPart, STORE[storeIdentifier]))
    g.add((STORE[storeIdentifier], BRICK.isPartOf, BLDG[storeName]))
    
    
    # g.add((STORE[storeIdentifier], BRICK.isPartOf, BLDG[buiOfStore]))
    # Definition of the area
    areaOfStore = [
        (BRICK.value, Literal(storeArea, datatype=XSD.double)),
        (BRICK.hasUnit, UNIT["M2"]),
    ]
    g.add((STORE[storeIdentifier], BRICK.area, areaOfStore))
    
    # LATITUDE AND LONGITUDE
    g.add((STORE[storeIdentifier], BRICK.latitude, lat))
    g.add((STORE[storeIdentifier], BRICK.longitude, long))
    
    
    #        HVAC zone of the Store
    g.add((STORE["HVAC_area"+storeIdentifier], A, BRICK.HVAC_Zone))
    # g.add((STORE["HVAC_area"+storeIdentifier], BRICK.isPartOf, STORE[storeIdentifier]))
    g.add((STORE[storeIdentifier], BRICK.hasPart, STORE["HVAC_area"+storeIdentifier]))
    
    #       SUBZONE of the HVAC ZONE
    # subzones = [
    #     {
    #         "zoneName":"Zone1"
    #         "temperature_sensor":""
    #     }
    # ]
    for n in range(numberHVACZone):
        g.add((STORE["Zone_"+str(n+1)], A, BRICK.Space))
        # g.add((STORE["Zone_"+str(n+1)], BRICK.isPartOf, STORE["HVAC_area"+storeIdentifier]))
        g.add((STORE["HVAC_area"+storeIdentifier], BRICK.hasPart, STORE["Zone_"+str(n+1)]))
        g.add((STORE["Temp_sensor_"+str(n+1)], A, BRICK.Air_Temperature_Sensor))
        # g.add((STORE["Temp_sensor_"+str(n+1)], BRICK.isPointOf, STORE["Zone_"+str(n+1)]))
        g.add((STORE["Zone_"+str(n+1)], BRICK.hasPoint, STORE["Temp_sensor_"+str(n+1)]))
    #     # ADD TELEMETRY TEMPERATURE
    
    
    # =================================================
    #           METERS ASSOCIATION 
    # =================================================   
    # ----------------------------------------
    #               BUILDING LEVEL
    # ----------------------------------------
    # add a full building meter 
    g.add((BLDG["building_electric_meter_"+buiOfStore], A, BRICK.Building_Electrical_Meter))
    g.add((BLDG[buiOfStore], BRICK.isMeteredBy, BLDG["building_electric_meter_"+buiOfStore]))

    # add sensors to the building meter
    # energy sensor
    g.add((BLDG["building_energy_sensor"], A, BRICK.Energy_Sensor))
    g.add((BLDG["building_electric_meter_"+buiOfStore], BRICK.hasPoint,BLDG["building_energy_sensor"] ))
    # g.add((BLDG["building_energy_sensor"], BRICK.isPointOf, BLDG["building_electric_meter_"+buiOfStore]))
    g.add((BLDG["building_energy_sensor"], BRICK.hasUnit, UNIT["KiloW-HR"]))
    # timeseries_props = [
    #     (BRICK.hasTimeseriesId, Literal("a7523b08-7bc7-4a9d-8e88-8c0cd8084be0"))
    # ]
    # g.add((BLDG["building_energy_sensor"], BRICK.timeseries, timeseries_props))

    # power sensor
    g.add((BLDG["building_power_sensor"], A, BRICK.Electric_Power_Sensor))
    g.add((BLDG["building_electric_meter_"+buiOfStore], BRICK.hasPoint, BLDG["building_power_sensor"]))
    # g.add((BLDG["building_power_sensor"], BRICK.isPointOf, BLDG["building_electric_meter_"+buiOfStore]))
    g.add((BLDG["building_power_sensor"], BRICK.hasUnit, UNIT["KiloW"]))
    # timeseries_props = [
    #     (BRICK.hasTimeseriesId, Literal("fd64fbc8-0742-4e1e-8f88-e2cd8a3d78af"))
    # ]
    # g.add((BLDG["building_power_sensor"], BRICK.timeseries, timeseries_props))

    # ----------------------------------------
    #               STORE LEVEL
    # ----------------------------------------
    # Add HVAC submeter
    submeters = [
        {
            "name": f"HVAC_electric_meter_{storeIdentifier}",
            "power_sensor_id": "647654d4-56ee-11ec-bf02-3dcb0419df3b_test",
            "energy_sensor_id": "647654d4-56ee-11ec-bf02-3dcb0419df3b_test2",
        }
    ]
    for submeter in submeters:
        g.add((BLDG[submeter["name"]], A, BRICK.Electrical_Meter))
        g.add((BLDG[submeter["name"]], BRICK.meters, STORE["HVAC_area"+storeIdentifier]))
        g.add((BLDG["building_electric_meter_"+buiOfStore], BRICK.hasSubMeter, BLDG[submeter["name"]]))
        # g.add((BLDG[submeter["name"]], BRICK.isSubMeterOf, BLDG["building_electric_meter_"+buiOfStore]))

        # electrical meter can provide power and energy
        # POWER
        g.add((BLDG["HVAC_power"+storeIdentifier], A, BRICK.Electric_Power_Sensor))
        # g.add((BLDG["HVAC_power_"+storeIdentifier], BRICK.isPointOf, BLDG[submeter["name"]]))
        g.add((BLDG[submeter["name"]], BRICK.hasPoint, BLDG["HVAC_power_"+storeIdentifier]))
        g.add((BLDG["HVAC_power_"+storeIdentifier], BRICK.hasUnit, UNIT["KiloW"]))
        timeseries_props = [(BRICK.hasTimeseriesId, Literal(submeter["power_sensor_id"]))]
        g.add((BLDG["HVAC_power_"+storeIdentifier], BRICK.timeseries, timeseries_props))

        # ENERGY
        g.add((BLDG["HVAC_energy_"+storeIdentifier], A, BRICK.Energy_Sensor))
        # g.add((BLDG["HVAC_energy_"+storeIdentifier], BRICK.isPointOf, BLDG[submeter["name"]]))
        g.add((BLDG[submeter["name"]], BRICK.hasPoint,BLDG["HVAC_energy_"+storeIdentifier] ))
        g.add((BLDG["HVAC_energy_"+storeIdentifier], BRICK.hasUnit, UNIT["KiloW-HR"]))
        timeseries_props = [(BRICK.hasTimeseriesId, Literal(submeter["energy_sensor_id"]))]
        g.add((BLDG["HVAC_energy_"+storeIdentifier], BRICK.timeseries, timeseries_props))

    return g




#%%
# ========================================================================
#                   GENERATE BRICK SCHEMA FOR ALL BUILDINGS 
# ========================================================================


generalInfo = pd.read_csv("../DATA/stores.csv", sep=";", decimal=",")

for i,building in generalInfo.iterrows(): 
    # building = pd.DataFrame(bui)
    inputs = {
        'storeIdentifier':building['Store identifier'],
        'storeName': building['Store name'],
        'storeArea':building['Store surface (mq)'],
        'latStore': building['Store latitude (decimal degrees)'],
        'longStore': building['Store longitude (decimal degrees)'],
        'numberHVACZone': building['Num HVAC/temperature areas']
    }
    print(i)
    print(inputs['storeIdentifier'])
    g = brick_data_model(**inputs)
    
g.serialize("wurth_shop.ttl", format="ttl")

#%%
# ========================================================================
#                   TEST SINGLE BUILDING
# ========================================================================
inputs = {
    'storeIdentifier':'BCFS',
    'storeName': 'Nola',
    'storeArea':541,
    'latStore': 14.46248,
    'longStore': 41.25925,
    'numberHVACZone': 4
    }

g = brick_data_model(**inputs)

g.serialize("wurth_shop_single.ttl", format="ttl")

#%%
# ================================================================
#                   VALIDATE THE MODEL
# ================================================================

g = brickschema.Graph(load_brick=True)
# OR use the absolute latest Brick:
# g = brickschema.Graph(load_brick_nightly=True)
# OR create from an existing model
# g = brickschema.Graph(load_brick=True).from_haystack(...)

# load in data files from your file system
g.load_file("wurth_shop.ttl")

# perform reasoning on the graph (edits in-place)
g.expand(profile="shacl")

# validate your Brick graph against built-in shapes (or add your own)
valid, _, resultsText = g.validate()
if not valid:
    print("Graph is not valid!")
    print(resultsText)
else: 
    print("Valid")
    

    






# %%
