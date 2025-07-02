import dash
from dash import get_relative_path, html, dcc
import  dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd
# dash.register_page(__name__,path=f"/processing")

name_DPR412 = [
    "E.1 (1) dwellings used for residence with continuous occupation, such as civil and rural dwellings, boarding schools, convents, penalty houses, barracks",
    "E.1 (2) dwellings used as residences with occasional occupation, such as vacation homes, weekend homes and the like",
    "E.1 (3) buildings used for hotel, boarding house and similar activities",
    "E.2 Office and similar buildings: public or private, independent or contiguous to buildings also used for industrial or craft activities, provided that they are separable from such buildings for the purposes of thermal insulation",
    "E.3 Buildings used as hospitals, clinics or nursing homes and assimilated including those used for the hospitalization or care of minors or the elderly as well as sheltered facilities for the care and recovery of drug addicts and other persons entrusted to public social services",
    "E.4 Buildings used for recreation or worship and similar activities",
    "E.4 (1) such as cinemas and theaters, conference meeting rooms",
    "E.4 (2) such as exhibitions, museums and libraries, places of worship",
    "E.4 (3) such as bars, restaurants, dance halls",
    "E.5 Buildings used for commercial and similar activities: such as stores, wholesale or retail warehouses, supermarkets, exhibitions",
    "E.6 Buildings used for sports activities",
    "E.6 (1) swimming pools, saunas and similar",
    "E.6 (2) gymnasiums and similar",
    "E.6 (3) support services for sports activities"
]

code_EPC = list(range(1,15))

df_categories = pd.DataFrame({
    "name_DPR412" : name_DPR412,
    "code_EPC" : code_EPC 
})
list_of_lists = [[row["name_DPR412"].split(" ", 1)[0] + row["name_DPR412"].split(" ", 1)[1], row["code_EPC"]] for _, row in df_categories.iterrows()]

characters_list = [
    {
        "id": "info_filtering_EPC",
        "image": "akar-icons:info",
        "label": "Info Filtering EPC",
        "description": dmc.Text("Definition of filters applied to the EPC Dataset", c="gray"),
        "content": dmc.Stack(
            children = [
                html.P("Below is shown the approach used to clean the dataset, considering different physical aspects of the building such that an EPC is considered valid. These filters derive from a study of different facts that geometrically and physically characterize a building, such that an analysis cannot be considered valid if it does not pass these filters.", style = {'marginBottom':'0px'}),
                html.P("Based on technical and regulatory criteria, and considering the characteristics of public buildings, these filters were crucial for identifying and removing incorrect or implausible data. Their application led to the removal of 1377 EPCs from the original dataset due to non-compliance with one or more criteria, leaving a more accurate and reliable dataset of 2001 valid EPCs. It's noteworthy that the total number of removed buildings exceeds 1377, as many failed multiple criteria, highlighting a general accuracy issue in EPC compilation, with about 41% of the original dataset discarded due to clear inconsistencies.", style = {'marginTop':'0px', 'marginBottom':'0px'}),
                dmc.List(
                    icon=dmc.ThemeIcon(
                        DashIconify(icon="radix-icons:check-circled", width=16),
                        radius="xl",
                        color="teal",
                        size=24,
                    ),
                    size="sm",
                    spacing="sm",
                    children=[
                        dmc.ListItem(
                            html.P( 
                                [
                                    html.B("INTERFLOOR: "), 
                                    "The inter-floor filter is designed to exclude EPCs with implausible inter-floor heights, indicating potential errors in the architectural data entered. This check is crucial to ensure that building dimensions are accurately represented, directly affecting calculations related to heated volume and, consequently, energy assessment. \
                                        To ensure the accuracy of the inter-floor height indicated in EPCs, a minimum limit of 2.8 meters has been established. This value reflects standard construction characteristics of public buildings, considering a typical minimum internal height for residences and adding 30 cm for floor thickness."
                                ],
                            )
                        ),
                        dmc.ListItem(
                            html.P( 
                                [
                                    html.B("MINIMUM THERMAL SYSTEM EFFICIENCY: "), 
                                    "A minimum efficiency filter is applied to exclude EPCs reporting thermal plant efficiencies below a defined threshold, indicating potential misclassification of their efficiency. This threshold ensures that heating systems in the dataset can operate efficiently. Key aspects considered for each system component:",
                                    html.Li(" - Combustion Efficiency (ηcombustione): Minimum legal is 87%;"),
                                    html.Li(" - Distribution Efficiency (ηdistribuzione): Acceptable minimum is around 85%."),
                                    html.Li(" - Emission Efficiency (ηemissione): Acceptable minimum is around 90%."),
                                    html.Li(" - Regulation Efficiency (ηregolazione): Acceptable minimum is around 85%."),
                                    "The overall heating system efficiency must exceed the product of these minimum sub-efficiencies:",
                                    html.Li(html.B("ηGlobalHeating > 0.87*0.85*0.90*0.85 (=0.5657)"))
                                ],
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("PRIMARY AND THERMAL ENERGY NEED - higer than 0: "),
                                    "This filter is designed to identify and exclude EPCs with significant data inconsistencies: buildings that have a thermal energy requirement but report zero energy consumption for heating. This suggests a potential error in the data, as it is unlikely for a building with a thermal requirement to have no associated heating energy consumption."
                                    "The filter operates with the following conditional logic:",
                                    html.Li(html.B("Qh>0 and EPh>0"))
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("MAXIMUM DISPERSIVE SURFACE: "),
                                    "The maximum dispersive surface filter is designed to identify and discard EPCs reporting unrealistically high dispersive surface values for the type of buildings considered, indicating potential measurement or recording errors. To determine an acceptable maximum dispersive surface value, a fictitious building concept is introduced. This model is designed with a width of 1.5 meters, similar to a corridor, consisting solely of the ground floor, and features a 30-degree pitched roof. \
                                        The comparison criterion is as follows: if the total dispersive surface area of the analyzed building exceeds that of the defined fictitious building, the EPC is considered unreliable and thus discarded. The maximum dispersive surface is calculated using the formula:",
                                    html.Li(html.B("[(Heated Surface/1.5) + 1.5]*2* interfloor_height + (1+ 1/cos(30))* useful_heated_surface > opaque_dispersive_surface + transparent_dispersive_surface" )),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("MINIMUM HEATING GENERATOR POWER: "),
                                    "This filter aims to eliminate EPCs that have insufficient overall heating system power to maintain thermal comfort given a specific external design temperature. The calculation is based solely on transmission losses through the building envelope, excluding ventilation losses. We adopted a design temperature of 0°C, which is conservative compared to the maximum design temperature of -5°C commonly used in Piedmont, to introduce a safety margin. \
                                        Our methodological decision not to consider ventilation losses and to use a higher design temperature ensures that only adequately sized systems are deemed suitable. Therefore, the filter discards those EPCs where the system's total power cannot compensate for transmission losses at an external temperature of 0°C, indicating that the system may not be appropriately sized for the building's thermal characteristics and surface area. \
                                        The formula used for this filter is: ",
                                    html.Li(html.B("opaque_dispersive_surface * average_transmittance_opaque_dispersive_surface + transparent_dispersive_surface * average_transmittance_transparent_dispersive_surface * ((20°C - 0°C)/1000) < Overall_Power_of_heating_system" )),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("AIR CHANGE PER HOUR"),
                                    "For public buildings, the hourly air exchange rate must be within a defined range to ensure effectiveness and efficiency.\
                                        The minimum rate, as per residential regulations, is 0.3 volumes per hour to maintain air quality. \
                                             This standard also applies to public buildings. The upper limit is determined by a maximum occupancy index of 1.5 people per square meter, \
                                                as per UNI 10339 for sports facilities. With a ventilation flow rate of 39.6 cubic meters per hour per person, \
                                                    the theoretical maximum hourly air exchange rate can be calculated. This formula determines the acceptable range for hourly air exchange.",
                                    html.Li(html.B("0.3 < air_change_per_hour < (maximum_occupancy_index * heated_useful_floor_area + 39.6)/building_volume")),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("DAYLIGHT AND NATURAL VENTILATION AREA: "),
                                    "The aerated area filter aims to eliminate EPCs that do not meet the minimum natural lighting requirements established for archives, warehouses, and unattended storage rooms. \
                                        These environments, characterized by naturally less stringent lighting requirements compared to other types of buildings, offer the least restrictive reference value for defining the filter. \
                                        The specific formula used to apply the aerated area filter is as follows:",
                                    html.Li(html.B("[transparent_heat_loss_surface/heated_useful_floor_area]> 1/30")),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("EXPOSED SOLAR SURFACE - Asol: "),
                                    "The Asol filter is designed to verify the consistency of the exposed solar surfaces relative to the orientation of buildings. The objective is to ensure that the information reported on the EPCs is realistic and accurately reflects the architectural characteristics of the buildings analyzed.\
                                        This filter eliminates EPCs that present incredible exposed solar surface (Asol) values, that is, when these surfaces exceed the total surface area of the windows. Such inconsistency could indicate registration errors or calculation mistakes related to the actual solar exposure of the buildings. \
                                            The formula used to apply this filter is as follows:",
                                    html.Li(html.B("Asol <= transparent_heat_loss_surface")),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("CONSTRUCTION YEAR vs HVAC CONSTRUCTION YEAR: "),
                                    "This filter was implemented to identify and discard EPCs that show temporal inconsistencies between the building construction year and the installation year of thermal or cooling systems. Such inconsistencies can suggest errors in the input data or the EPC records, compromising the validity of the energy analysis.\
                                    The filter ensures that the building construction year is consistent with or precedes the installation year of the systems, allowing a tolerance margin of 3 years. This margin was introduced to consider the possibility that the generator's construction date might have been erroneously recorded as the system's construction date. This verification is crucial to guarantee the plausibility of the information, as the systems cannot be installed before the building itself is constructed. \
                                        The formula applied for this filter is as follows:",
                                    html.Li(html.B("building_construction_year <= hvac_construction_year + 3")),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("MINIMUM HEATED SURFACE AREA: "),
                                    "This filter has been implemented to exclude EPCs that report heated areas too small to ensure adequate comfort and functionality of the building, based on its specific intended use. Each building category has minimum heated surface area requirements, determined according to the operational needs and comfort of the occupants. Below are the specific criteria for various types of buildings:",
                                    html.Li([html.B("Residential Buildings - Continuous dwellings (residential houses, colleges, convents, penitentiaries, barracks)"),"The minimum considered area is 80 m² for structures with multiple rooms and common spaces."]), 
                                    html.Li([html.B("Residential Buildings - Occasional occupancy dwellings (holiday homes) "), "The minimum area for a studio apartment is 40 m² for slightly larger structures."]), 
                                    html.Li([html.B("Hotels and similar structures: "), "A small hotel structure should cover at least 100 m², including rooms and common areas."]), 
                                    html.Li([html.B("Office: "), "A small office should measure at least 40 m², including individual office spaces and common areas"]), 
                                    html.Li([html.B("Health facilities (hospitals, clinics): "), "The minimum area for a small clinic is estimated at 200 m², considering rooms and common areas"]), 
                                    html.Li([html.B("Recreational or worship facilities (cinemas, theaters, museums, churches): "), "The minimum area varies from 80 m² for small exhibition structures."]), 
                                    html.Li([html.B("Bars, restaurants, and dance halls: "), "Minimum 50 m²"]), 
                                    html.Li([html.B("Commercial activities: "), "A small shop should have an area of at least 30 m²."]), 
                                    html.Li([html.B("Sports activities (swimming pools, gyms): "), "The minimum area starts from 100 m² for a small gym."]), 
                                    html.Li([html.B("Educational institutions: "), "A small school should have at least 80 m², including classrooms and common areas."]), 
                                    html.Li([html.B("Industry and crafts: "), "The minimum area for a workshop or a small craft factory is 100 m²"]), 
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("WINDOW TRANSMITTANCE: "),
                                    "This filter has been designed to identify and discard EPCs that report unrealistic or non-compliant thermal transmittance values for glass (Uwindow).The filter excludes:",
                                    html.Li("- EPCs with thermal transmittance values lower than what would be obtained with a modern window equipped with triple low-emissivity glass, whose typical value is less than 0.6 W/(m²K)."),
                                    html.Li("- EPCs with thermal transmittance values higher than what would be obtained with an obsolete window with single glass and aluminum frame without thermal break, typically higher than 8 W/(m²K)."),
                                    html.Li("The formula used to apply this filter is:"),
                                    html.Li(html.B("0.6 ≤ Window_Transmittance ≤ 8")),
                                ]
                            )
                        ),
                        dmc.ListItem(
                            html.P(
                                [
                                    html.B("OPAQUE AND TRANSPARENT SURFACE:"),
                                    html.Li(html.B("Default filter that remove buildings in which the opaque surface and transparent surface should be higher than 0")),
                                ]
                            )
                        ),
                    ],
                )
            ]
        )
        
    },
]

def create_accordion_label(label, image, description):
    return dmc.AccordionControl(
        dmc.Group(
            [
                DashIconify(icon=image, width=30, color="rgb(121, 80, 242)"),
                html.Div(
                    [
                        dmc.Title(label, order=3),
                        description, 
                    ]
                ),
            ]
        )
    )


def create_accordion_content(content):
    return dmc.AccordionPanel(dmc.Text(content, size="sm"))

filters_name = [
    {'label':'INTERFLOOR', 'checked':False},
    {'label':'MINIMUM THERMAL SYSTEM EFFICIENCY', 'checked':False},
    {'label':'PRIMARY AND THERMAL ENERGY NEED', 'checked':False},
    {'label':'MAXIMUM DISPERSIVE SURFACE', 'checked':False},
    {'label':'MINIMUM HEATING GENERATOR POWER', 'checked':False},
    {'label':'EXPOSED SOLAR SURFACE', 'checked':False},
    {'label':'CONSTRUCTION YEAR vs HVAC CONSTRUCTION YEAR', 'checked':False},
    {'label':'MINIMUM HEATED SURFACE AREA', 'checked':False},
    {'label':'WINDOW TRANSMITTANCE', 'checked':False},
    {'label':'AIR CHANGE PER HOUR', 'checked':False},
    {'label':'DAYLIGHT AND NATURAL VENTILATION AREA', 'checked':False},
    {'label':'EXPOSED SOLAR SURFACE', 'checked':False},
]

filters = dmc.Fieldset(
    children = [
        dmc.Checkbox(label = "ALL FILTERS", id="all_filters", variant="outline", checked=True),
        dmc.Divider(variant="solid", size="md", c="black", mt=10, mb=10),
        dmc.Group(
            children = [
                dmc.Checkbox(
                    id={"type":"notification-item", "index":i},
                    label = item["label"],
                    variant="outline",
                    checked = item["checked"],
                )
                for i, item in enumerate (filters_name)  
            ],
            justify="space-between"
        )
    ],
    legend=dmc.Text("Filters", fw=700, size="xl"),
)

data = [
    {"month": "January", "Smartphones": 1200, "Laptops": 900, "Tablets": 200},
    {"month": "February", "Smartphones": 1900, "Laptops": 1200, "Tablets": 400},
    {"month": "March", "Smartphones": 400, "Laptops": 1000, "Tablets": 200},
    {"month": "April", "Smartphones": 1000, "Laptops": 200, "Tablets": 800},
    {"month": "May", "Smartphones": 800, "Laptops": 1400, "Tablets": 1200},
    {"month": "June", "Smartphones": 750, "Laptops": 600, "Tablets": 1000}
]

graph_1 = dmc.Paper(
    children = [
        dmc.Group(
            children = [
                dmc.Title(children= "Number of EPCS after the application of filters:", order=4),
                dmc.Title(id="number_of_epc", order=3, c="violet"),
            ]
        ),
        dmc.Divider(variant="solid", mt=10, mb=10, size="md", c="grey"), 
        dmc.Grid(
            children = [
                dmc.GridCol(
                    children = [
                        dmc.Select(
                            id="id_variable",
                            label = "Energy parameters",
                            description="List of variables",
                            data = [
                                {"label": "Total opaque surface", "value": "total_opaque_surface"},
                                {"label": "Total glazed surface", "value": "total_glazed_surface"},
                                {"label": "Heated net area", "value": "heated_usable_area"},
                                {"label": "Cooled net area", "value": "cooled_usable_area"},
                                {"label": "Heated gross volume", "value": "heated_gross_volume"},
                                {"label": "Cooled gross volume", "value": "cooled_gross_volume"},
                                {"label": "Average opaque surface transmittance", "value": "average_opaque_surface_transmittance"},
                                {"label": "Average glazed surface transmittance", "value": "average_glazed_surface_transmittance"},
                                {"label": "Exposed solar surface", "value": "assembled_solar_surface"},
                                {"label": "Nominal power", "value": "nominal_power"},
                                {"label": "Primary Energy for Heating", "value": "EPh"},
                                {"label": "Ideal useful thermal energy requirement (Qh,nd) per unit of surface/volume", "value": "QHnd"},
                                {"label": "Net area", "value": "net_area"},
                                {"label": "air_changes", "value": "air_changes"},
                            ],
                            value = "total_opaque_surface"
                        ),
                        dmc.Divider(variant="solid", color="gray", size="sm", mt=20, mb=10),
                        dmc.Center(dmc.Title("BUILDING CATEGORY OF EPCs", order=4, mt=10, mb=10, c="violet")),
                        dmc.ScrollArea(
                            children = dmc.Table(
                                data={
                                    "caption": "Building category according to Italian DPR412",
                                    "head": ["DPR 412", "code_EPC"],
                                    "body": list_of_lists,
                                }
                            ),
                            type="always",
                            offsetScrollbars=True,
                            h=350,
                            mt=10
                        )
                    ],
                    span=4
                ),
                dmc.GridCol(
                    children = [
                        html.Div(id="graph_processing_1")
                    ],
                    span=8
                ),

            ],
            mt=10
        )
    ],
    radius="md",
    shadow="md",
    p=10
)

accordions = dmc.Accordion(
    chevronPosition="right",
    variant="contained",
    children=[
        dmc.AccordionItem(
            [
                create_accordion_label(
                    character["label"], character["image"], character["description"]
                ),
                create_accordion_content(character["content"]),
            ],
            value=character["id"],
        )
        for character in characters_list
    ],
)

# main = [
#     dcc.Store(id="data_filtered", storage_type="session"),
#     accordions, 
#     dmc.Divider(variant = "solid", color="lightgrey", size="md", mt=10, mb=10),
#     filters,
#     graph_1,
#     clustering_analysis
# ]

# ===========================================================================
#                               LAYOUT MAIN
# ===========================================================================

# layout_ = html.Div(
#     children = dmc.Container(
#         size="xl",
#         children = main)
# )

# def layout():
#     return layout_