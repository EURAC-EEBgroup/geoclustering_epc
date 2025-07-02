import dash
from dash import get_relative_path, html, dcc
import  dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd



elbow_and_silhoutte_graph = html.Div(
    children = [
        dmc.Grid(
            children = [
                dmc.GridCol(
                    children = [
                        html.Div(id="elbow_graph")
                    ],
                    span={"base": 12, "md": 6, "lg":6}
                ),
                dmc.GridCol(
                    children = [
                        html.Div(id="silhouette_graph")
                    ],
                    span={"base": 12, "md": 6, "lg":6}
                )
            ]
        )
    ]
)

clustering_analysis = dmc.Paper(
    children = [
        dmc.Grid(
            children=[
                dmc.GridCol(
                    children = [
                        dmc.Select(
                            label = "Building typology",
                            id="building_type",
                            data = [
                                {"label": "Office", "value": "4"},
                                {"label": "Hospital", "value": "5"},
                                {"label": "Building used for recreation or worship", "value": "6"},
                                {"label": "gymnasiums and similar", "value": "13"},
                                {"label": "Buildings used for commercial and similar activities", "value": "10"},
                                {"label": "Buildings used for sports activities", "value": "11"},
                                {"label": "Swimming pools, Saunas and Similar", "value": "12"},
                                {"label": "Exhibitions, museums and libraries", "value": "8"},
                                {"label": "Cinema and theaters, conference meeting rooms", "value": "7"},
                            ],
                            value="4",
                            mt=10,
                        ),
                        dmc.MultiSelect(
                            id="parameters_cluster",
                            label = "Parameters for clustering",
                            data = [
                                {"label": "Heating energy need - QHnd", "value": "QHnd"},
                                {"label": "Primary energy - kwh/m2 year", "value": "EPh"},
                                {"label": "Degree days", "value": "degree_days"},
                                {"label": "Heat loss surface", "value": "heat_loss_surface"},
                                {"label": "Heated usable area", "value": "heated_usable_area"},
                                {"label": "Average opaque surface transmittance", "value": "average_opaque_surface_transmittance"},
                                {"label": "Average glazed surface transmittance", "value": "average_glazed_surface_transmittance"},
                                {"label": "Surface to volume ratio", "value": "surface_to_volume_ratio"},
                                {"label": "Exposed solar surface", "value": "Asol"},
                                {"label": "Average transmittance of facade", "value": "average_U_facade"},
                            ],
                            value=["QHnd", "degree_days"],
                            maxValues=2,
                            mt=10,
                        ),
                        dmc.Checkbox(
                            id="Elbow_method",
                            label="Elbow method",
                            description="method to find the optimal number of clusters",
                            variant="outline",
                            checked=True,
                            mt=10,
                        ),
                        dmc.Checkbox(
                            id="Silhouette_method",
                            label="Silhouette method",
                            description="method to find the optimal number of clusters",
                            variant="outline",
                            checked=True,
                            mt=10,
                        ),
                        dmc.Select(
                            id="cluster_type",
                            label="Number of cluster",
                            data=[
                                {'label':'Best option', 'value':'best_option'},
                                {'label':'Custom number of cluster', 'value':'custom_number'}
                            ],
                            value="best_option",
                            mt=10,
                        ),
                        dmc.Select(
                            data = [
                                {'label':'silhouette', 'value':'silhouette'},
                                {'label':'elbow', 'value':'elbow'}
                            ],
                            label="Cluster method",
                            id="cluster_method",
                            value="silhouette",
                            mt=10,
                            disabled=True,
                        ),
                        dmc.NumberInput(
                            id="custom_number",
                            label="Custom number of cluster",
                            min=1,
                            mt=10,
                            value=2,
                            disabled=True,
                        ),
                        dmc.Button("Cluster", id="cluster_btn", variant="outline", fullWidth=True, radius="md", color="violet", mt=10, mb=10)
                    ], 
                    p=10,
                    span={"base": 12, "md": 5, "lg":4}),
                dmc.GridCol(
                    id="map_clustering", span={"base": 12, "md": 7, "lg":8}),
            ],
        ),
        elbow_and_silhoutte_graph,
        html.Div(id="cluster_analysis_graphs")
    ],
    radius="md",
    shadow="md",
    p=10
)


prediction = dmc.Stack(
    children = [
        dmc.Grid(
            children = [
                dmc.GridCol(
                    children = [
                        dmc.Alert(
                            id="alert_prediction",
                            children = [
                                "Before making a prediction generate the building clusters from the previous section"
                            ],
                            color="red",
                            variant="outline",
                            mt=10,
                            mb=10
                        ),
                        dmc.Select(
                            id="cluster_select",
                            label="Cluster",
                            mt=10,
                        ),
                        dmc.Select(
                            id="target_predict",
                            label="Target",
                            data=[
                                {'label':'Heating Energy Need - QHnd', 'value':'QHnd'},
                            ],
                            value="QHnd",
                            mt=10,
                        ),
                        dmc.MultiSelect(
                            id="parameters_prediction",
                            label="Select parameters",
                            data = [
                                {"label": "Average transmittance of opaque components", "value": "average_opaque_surface_transmittance"},
                                {"label": "Average transmittance of transparent components", "value": "average_glazed_surface_transmittance"},
                            ],
                            value=["average_opaque_surface_transmittance", "average_glazed_surface_transmittance"],
                            maxValues=2,
                            mt=10,
                        ),
                        dmc.Checkbox(
                            id="normalize",
                            label="Visualize sensitivity analysis",
                            description="evaluate how each parameter affect the target",
                            variant="outline",
                            checked=True,
                            mt=10,
                        ),
                        dmc.Checkbox(
                            id="scanrios",
                            label="Create scenarios to be evaluated",
                            description="create multiple scnarios to be evaluated",
                            variant="outline",
                            checked=True,
                            mt=10,
                        ),
                        dmc.Fieldset(
                            children = [
                                dmc.Grid(
                                    children = [
                                        dmc.GridCol(
                                            children = [
                                                dmc.NumberInput(
                                                    id="min_average_opaque_surface_transmittance",
                                                    label="Minimum Average transmittance of opaque components",
                                                    min=0.2,
                                                    max=1,
                                                    step=0.1,
                                                    value=0.5,
                                                    mt=10,
                                                ),
                                            ],
                                            span={"base": 12, "md": 12, "lg":6}
                                        ),
                                        dmc.GridCol(
                                            children = [
                                                dmc.NumberInput(
                                                    id="maximum_average_opaque_surface_transmittance",
                                                    label="Maximum Average transmittance of opaque components",
                                                    min=0.2,
                                                    max=1,
                                                    step=0.1,
                                                    value=1,
                                                    mt=10,
                                                ),
                                            ],
                                            span={"base": 12, "md": 12, "lg":6}
                                        ),
                                    ]
                                )
                            ],
                            legend = "Average transmittance of opaque components",
                            id="scanrios_definition"
                        ),
                        dmc.Fieldset(
                            children = [
                                dmc.Grid(
                                    children = [
                                        dmc.GridCol(
                                            children = [
                                                dmc.NumberInput(
                                                    id="min_average_glazed_surface_transmittance",
                                                    label="Minimum Average transmittance of glazed components",
                                                    min=0.2,
                                                    max=1,
                                                    step=0.1,
                                                    value=0.2,
                                                    mt=10,
                                                ),
                                            ],
                                            span={"base": 12, "md": 12, "lg":6}
                                        ),
                                        dmc.GridCol(
                                            children = [
                                                dmc.NumberInput(
                                                    id="maximum_average_glazed_surface_transmittance",
                                                    label="Maximum Average transmittance of glazed components",
                                                    min=0.2,
                                                    max=1,
                                                    step=0.1,
                                                    value=0.7,
                                                    mt=10,
                                                ),
                                            ],
                                            span={"base": 12, "md": 12, "lg":6}
                                        ),
                                    ]
                                )
                            ],
                            legend = "Average transmittance of transparent components",
                            id="scanrios_definition"
                        ),
                        dmc.Button("Predict", id="predict_btn", variant="outline", fullWidth=True, radius="md", color="violet", mt=10, mb=10, disabled=True)
                    ],
                    span={"base": 12, "md": 5, "lg":4}
                ),
                dmc.GridCol(
                    children = [
                        dmc.Stack(
                            children = [
                                dmc.Skeleton(id="skeleton_map_prediction", children = html.Div(id="map_prediction"),height="70vh", visible=False),
                                html.Div(id="sensitivity_analysis_graphs"),
                                html.Div(id="scenarios_graphs"),
                                
                            ]
                        )
                        
                    ],span={"base": 12, "md": 7, "lg":8}
                ),
                
            ],
        ),
        dmc.Grid(
            children = [
                dmc.GridCol(
                    children = [
                        html.Div(id="3d_influence"),
                    ],
                    span={"base": 12, "md": 6, "lg":6}
                ),
                dmc.GridCol(
                    children = [
                        html.Div(id="heat_map_influence"),
                    ],
                    span={"base": 12, "md": 6, "lg":6}
                ),
            ]
        )
    ]
)

