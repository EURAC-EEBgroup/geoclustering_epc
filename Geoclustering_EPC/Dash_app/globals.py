import os
import pandas as pd 
import requests
from dotenv import load_dotenv
load_dotenv()

directory_= os.getcwd()

BUILDING_RELATIVE_PATH = '/epc_clustering/piemonte'
DASH_URL_BASE_PATHNAME = '/epc_clustering/piemonte/'
os.environ['DASH_URL_BASE_PATHNAME'] = DASH_URL_BASE_PATHNAME
BUILDON_SECRET = os.getenv('BUILDON_SECRET', 'dmihUYAwgqXKRy3')

# df = pd.read_csv("Dash_app/data/dataset_EPC_cleaned.csv", sep=",", decimal=".", low_memory=False, header=0)
df = pd.read_csv("data/dataset_EPC_cleaned.csv", sep=",", decimal=".", low_memory=False, header=0)
df = df.loc[:, ['classificazione_DPR412', 'anno_costruzione', 'latitude', 'longitude', 'gradi_giorno', 'altitudine', 'floors', 'superficie_Netta', 'superficie_Disperdente', 'superficie_Utile_Riscaldata', 'superficie_Utile_Raffrescata', 'volume_Lordo_Riscaldato', 'volume_Lordo_Raffrescato', 'superficie_Opaca_Totale', 'superficie_Vetrata_Totale', 'superficie_Opaca_Trasmittanza_Media', 'superficie_Vetrata_Trasmittanza_Media', 'yie', 'Cm', 'rapportoSV', 'Asol', 'tipo_Impianto', 'potenza_Nominale', 'anno_Installazione', 'EPh', 'QHnd', 'QHimp', 'portata_Ventilazione_Effettiva_Totale', 'tipologia_Ventilazione', 'ricambi_Aria', 'EPv', 'EPc', 'EPl', 'EPt', 'EPgl', 'EPw']].dropna()
df.columns = ['DPR412_classification', 'construction_year', 'latitude', 'longitude', 'degree_days', 'altitude', 'floors', 'net_area', 'heat_loss_surface', 'heated_usable_area', 'cooled_usable_area', 'heated_gross_volume', 'cooled_gross_volume', 'total_opaque_surface', 'total_glazed_surface', 'average_opaque_surface_transmittance', 'average_glazed_surface_transmittance', 'yie', 'Cm', 'surface_to_volume_ratio', 'Asol', 'system_type', 'nominal_power', 'installation_year', 'EPh', 'QHnd', 'QHimp', 'total_effective_ventilation_flow', 'ventilation_type', 'air_changes', 'EPv', 'EPc', 'EPl', 'EPt', 'EPgl', 'EPw']
df = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
df = df[~df.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
df = df.reset_index(drop=True)

# path_result="Dash_app/result_sensitivity_cluster"
path_result="result_sensitivity_cluster"