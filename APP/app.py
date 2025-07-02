import os 


import dash_mantine_components as dmc
import dash 
from dash import Dash, html,dcc,Output, Input, State, callback,ctx, ALL
from dash import get_relative_path, strip_relative_path
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

from flask import Flask, request, redirect, session
from flask_login import login_user, LoginManager, UserMixin, logout_user, current_user
from werkzeug.security import check_password_hash
from dash import DiskcacheManager, CeleryManager, Input, Output, html
# from callbacks import callbacks_geoclustering, callback_nuts, callback_benchmarking, callback_drawer, callback_admin, \
#     callback_user_reg

# ===============================================================

server = Flask(__name__)# root_path=GEOAPP_RELATIVE_PATH)
#
app = dash.Dash(
    __name__,
    server = server,
    meta_tags=[
        {
            'charset': 'utf-8',
        },
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1, shrink-to-fit=no'
        }
        ],    
    use_pages=True,
    assets_folder='assets',
    # background_callback_manager=background_callback_manager,
    # suppress_callback_exceptions=False
    # long_callback_manager=long_callback_manager
    )

app.config.suppress_callback_exceptions = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.title = 'Benchmarking'

server = app.server

# ===============================================================
#                       LAYOUT
# ===============================================================
app.layout = html.Div(
    children = [
        html.Div(
            [
                dcc.Location(id='url_app'),
                html.Div(id="user-status-header"),
                dcc.Store('current_user_id', storage_type="session"),
                dash.page_container
            ],
        )
    ],
    style = {'backgroundColor': '#E8EDF1',
             'zoom':'90%'} 
)


#  ==============================================================================================================================

if __name__ == '__main__':
    app.run_server(port=8082, debug=True, dev_tools_ui=True,dev_tools_props_check=False)
    
  

