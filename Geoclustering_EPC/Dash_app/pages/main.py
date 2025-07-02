import dash
from dash import html, dcc
import  dash_mantine_components as dmc
from dash_iconify import DashIconify
from pages.processing import list_of_lists, accordions, filters, graph_1
from pages.analysis import data_table, plots_univariate_distribution, correlation_matrix, info_1
from pages.clustering import clustering_analysis, prediction

dash.register_page(__name__,path=f"/")

# ===========================================================================
#                               ANALYSIS
# ===========================================================================

initial_center = [45.02569105418987, 7.671092180850915]
main_analysis = html.Div(
    id="prj_card_analysis",
    children = [
        dmc.Stack(
            
            children = [
                dmc.Group(
                    children = [
                        DashIconify(icon="streamline:code-analysis-solid",color=dmc.DEFAULT_THEME["colors"]["gray"][6],width=72),
                        dmc.Stack(
                            children = [
                                dmc.Title("Analysis", order=1),
                                dmc.Title("Analysis of EPCs", order=3, c="#dee2e6")
                            ],
                            gap=1
                        )
                    ]
                ),
                data_table,
                dmc.Text("Building map", size="xl", fw=800, mb=10, mt=5),
                dmc.Grid(
                    children=[
                        dmc.GridCol(
                            
                            dmc.Select(
                                id="map_inputs", 
                                label = "Variable",
                                data = [
                                    {"label": "building typology", "value": "DPR412_classification"},
                                    {"label": "construction year", "value": "construction_year"},
                                ],
                                value = "construction_year",
                                clearable = False,
                                radius="md",
                            ),
                            span=3
                        ),
                        dmc.GridCol(
                            id="map_bui",
                            span=9
                        )
                    ],
                    mt=10,
                    mb=10
                ),
                dmc.Text("Univariate Distribution of variable", size="xl", fw=800, mb=10, mt=5),
                plots_univariate_distribution,
                dmc.Text("Classification buildng type", size="xl", fw=800, mb=10, mt=5),
                dmc.Grid(
                    children = [
                        dmc.GridCol(
                            children = dmc.ScrollArea(
                                children = dmc.Table(
                                    data={
                                        "caption": "Building category according to Italian DPR412",
                                        "head": ["DPR 412", "code_EPC"],
                                        "body": list_of_lists,
                                    }
                                ),
                                type="always",
                                offsetScrollbars=True,
                                h=500,
                                mt=10
                            ),
                            span=6
                        ),
                        dmc.GridCol(
                            children = info_1,
                            span=6
                        )
                    ]
                ),
                dmc.Text("Correlation Matrix", size="xl", fw=800, mb=10, mt=5),
                correlation_matrix,
                
            ]
        )       
    ]
)

# ===========================================================================
#                               PROCESSING
# ===========================================================================

main_processing = html.Div(
    id="prj_card_processing",
    children = [
        dmc.Group(
            children = [
                DashIconify(icon="fluent-mdl2:processing",color=dmc.DEFAULT_THEME["colors"]["gray"][6],width=72),
                dmc.Stack(
                    children = [
                        dmc.Title("Processing", order=1),
                        dmc.Title("Processing of EPCs", order=3, c="#dee2e6")
                    ],
                    gap=1
                )
            ],
            mt=50,
            mb=20
        ),
        dcc.Store(id="data_filtered", storage_type="session"),
        accordions, 
        dmc.Divider(variant = "solid", color="lightgrey", size="md", mt=10, mb=10),
        filters,
        graph_1,
        
    ]
)

# ===========================================================================
#                               CLUSTERING AND PREDICTION
# ===========================================================================
main_clustering = html.Div(
    id="prj_card_clustering",
    children = [
        dcc.Store(id="data_clustered", storage_type="session"),
        dcc.Store(id="number_of_cluster", storage_type="session"),
        dcc.Store(id="test", storage_type="session"),
        dmc.Group(
            children = [
                DashIconify(icon="carbon:assembly-cluster",color=dmc.DEFAULT_THEME["colors"]["gray"][6],width=72),
                dmc.Stack(
                    children = [
                        dmc.Title("Clustering", order=1),
                        dmc.Title("Clustering of EPCs", order=3, c="#dee2e6")
                    ],
                    gap=1
                )
            ],
            mt=20,
            mb=20
        ),
        clustering_analysis,
        dmc.Group(
            children = [
                DashIconify(icon="carbon:assembly-cluster",color=dmc.DEFAULT_THEME["colors"]["gray"][6],width=72),
                dmc.Stack(
                    children = [
                        dmc.Title("Potential energy reduction", order=1),
                        dmc.Title("Analysis of building stock using EPCs", order=3, c="#dee2e6")
                    ],
                    gap=1
                )
            ],
            mt=20,
            mb=20
        ),
        dmc.Alert(
            "Here is an example of calculation with a model already generated and loaded in session that analyzes the clusters for the building energy demand, realized taking into account as parameter the degrees day. \
            The number of clusters is 4 obtained with Elbow method. And for the most numerous cluster a sensitivity analysis is performed to understand how the variables relative to the thermal transmittance of the envelope and the glazed surfaces can influence the consumption varying their values",
            title="Attention",
            color="red",
            mb=20,
            mt=20
        ),
        prediction
    ]
)


# ===========================================================================
#                               GENERAL
# ===========================================================================

layout_general= html.Div(
    children = [
        dmc.AppShell(
            [
                dcc.Loading([],
                    custom_spinner = html.Span(className= "loader_spin_2"),
                    style={'marginTop':'0px', 'marginBottom':'10px'},
                    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                ),
                dmc.AppShellMain(
                    children=[
                        dmc.Container(
                            size="xl",
                            children = [
                                main_analysis,
                                main_processing,
                                main_clustering,
                                html.Div(id="test"),
                                html.Div(id="clientside_callback_output")
                            ],
                            style = {'scrollBehavior': "smooth"}
                        )
                        ],
                    p=30
                ),
                
            ]
        ),
    ],
    style = {'marginBottom':'20px'}
)



def layout():
    return layout_general

