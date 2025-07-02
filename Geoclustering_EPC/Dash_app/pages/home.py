import dash
from dash import get_relative_path, html, dcc
import  dash_mantine_components as dmc
from dash_iconify import DashIconify
import flask
# from components.footer import Footer_home
from components.header import Header_home

# dash.register_page(__name__,path="/")

main_ = dmc.Container(
    size="xl",

)

# ==================================================================================================================
#                                                LAYOUT OVERALL
# ==================================================================================================================

layout_home = html.Div(
    children = [
        dmc.AppShell(
            [
                # Header_home,
                dcc.Loading(
                    [
                        # dcc.Store(id="data_energy_home", storage_type="session"),
                        # dcc.Store(id="energy_monthly_daily_average", storage_type="session"),
                        # dcc.Store(id="data_temperature_home", storage_type="session"),
                        # dcc.Store(id="data_co2_home", storage_type="session"),
                    ],
                    custom_spinner = html.Span(className= "loader_spin_2"),
                    style={'marginTop':'0px', 'marginBottom':'10px'},
                    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                ),
                dmc.AppShellMain(
                    children=[main_],
                    # style = {'backgroundColor': '#f1f3f5'}
                ),
            ],
            # style = {'backgroundColor': '#f1f3f5'}
        ),
        # Footer_home
    ],
    style = {'marginBottom':'20px'}
)

def layout():
    if not flask.session.get('authenticated'):
        return dcc.Location(href=get_relative_path("/login"), id="redirect-login")
    return layout_home