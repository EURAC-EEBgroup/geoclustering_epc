import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

# ==============================================================================
#                           LEFT DRAWER ADMIN 
# ==============================================================================
DrawerRight = dmc.Drawer(  
    title= [
        dmc.Text(
            "ANALYSIS",
            style = {'fontWeight':800,
                     'color':'black'}
        )
    ],
    children = [
        dmc.Divider(variant="solid", color = "black"),
        html.Br(),
        dmc.Text(
            "CLUSTER ANALYSIS",
            style = {
                'fontWeight': 600, 'color':'black'
            }),     
        # dmc.Select(
        #     id = "KPIS_admin_nuts",
        #     data = [
        #         {'value':1, 'label':'Energy Saving Normalized [kWh/m2HDD]'},
        #         {'value':2, 'label':'Heating consumption before refurbishment [kWh/m2]'},
        #         {'value':3, 'label':'Heating consumption after renovation [kWh/m2]'},
        #         {'value':4, 'label':'Renovation_cost [euro]'},
        #         {'value':5, 'label':'PayBack time of the investment(YEAR)'},
        #         {'value':6, 'label':'Cost_Saving [euro]'},
        #     ],
        #     # label="KPIs",
        #     value = 1,
        #     icon=DashIconify(icon="gis:road-map", width=25),
        #     rightSection=DashIconify(icon="radix-icons:chevron-down", color = "black"),
        # ),

        dmc.Button(
            "CLUSTERING",
            id = "btn_building_map",
            fullWidth=True
        )
    ],                          
    position="right",
    id="drawerRight",
    padding="md",
    withCloseButton = False,                
)