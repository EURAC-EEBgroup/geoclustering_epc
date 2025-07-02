import os
import dash_mantine_components as dmc
from flask import Flask, session
import dash
from dash import dcc, html, Input, Output, State, callback, get_relative_path, clientside_callback
from globals import BUILDING_RELATIVE_PATH
from flask_login import LoginManager, UserMixin
import ast  
from dotenv import load_dotenv
from components.footer import Footer
load_dotenv()

# ================================================================================
#                           GENERAL SETTINGS
# ================================================================================

# dash mantine components >= 14.0.1 requires React 18+
dash._dash_renderer._set_react_version("18.2.0")
server = Flask(__name__, root_path=BUILDING_RELATIVE_PATH)

external_stylesheets = [
    dmc.styles.DATES,
    dmc.styles.CAROUSEL,   
    dmc.styles.CHARTS,
    dmc.styles.NOTIFICATIONS,
    dmc.styles.CHARTS,
    "https://unpkg.com/@mantine/core@7.4.2/styles.css",
    "https://unpkg.com/@mantine/core@7.4.2/styles.layer.css"
]


# Login Manager Setup
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'
# User Class
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.authenticated = False
    
    def is_authenticated(self):
        return self.authenticated
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return self.id


@login_manager.user_loader
def load_user(user_id):
    if user_id not in USERS:
        return None
    user = User(user_id)
    return user


app = dash.Dash(
    __name__, 
    server=server, 
    use_pages=True,
    assets_folder='assets',
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets
)
app.title = 'bench'
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server = app.server
server.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', os.urandom(12)),
    SESSION_COOKIE_NAME='buildon_benchmark'
)


# ================================================================================
#                           LAYOUT  
# ================================================================================
from callbacks import callback_login, callback_home, callback_processing, callback_analysis, callback_synthetic, callback_clustering
from components.header import Header
# from components.navbar import Navbar
from dash_iconify import DashIconify

theme_toggle = dmc.Switch(
    offLabel=DashIconify(
        icon="radix-icons:sun", width=15, color=dmc.DEFAULT_THEME["colors"]["yellow"][8]
    ),
    onLabel=DashIconify(
        icon="radix-icons:moon",
        width=15,
        color=dmc.DEFAULT_THEME["colors"]["yellow"][6],
    ),
    id="color-scheme-toggle",
    persistence=True,
    color="grey",
)

# ICONS = {
#     "Home":"solar:home-line-duotone", 
#     "Analysis":"icon-park-twotone:market-analysis",
#     "Processing":"fluent-mdl2:processing",
#     "Synthetization":"solar:copy-bold-duotone",
#     "Clustering":"vaadin:cluster",
# } 


# navlinks = []
# for i, page in enumerate(dash.page_registry.values()):
#     if page["name"] != "Login":
#         navlinks.append(dmc.NavLink(
#             label=dmc.Text(f"{page['name']}", style={"fontSize": 15}, fw=500, ml=2),
#             # leftSection=DashIconify(icon=ICONS[page["name"]], width=18, color="rgb(121, 80, 242)"),
#             href=page["relative_path"],
#             id={"type": "navlink_", "index": page["relative_path"]},
#             variant="light",
#             autoContrast=False,
#             style = {'borderRadius': '10px'}
#         ))


content_navbar = [
    # Analysis
    dmc.Center(
        children = [
            dmc.ActionIcon(
                dmc.Tooltip(
                    label = "Analysis",
                    position="left",
                    offset=3,
                    transitionProps={
                        "duration": 300,
                        "easing": "ease-in-out",
                        "transition":"scale-x"
                    },
                    radius ='10px',
                    color=dmc.DEFAULT_THEME["colors"]["violet"][6],
                    children = [DashIconify(icon="gis:home", width=20, color=dmc.DEFAULT_THEME["colors"]["violet"][6]),]
                ),
                id="btnAnalysis",
                size="xl",
                color=dmc.DEFAULT_THEME["colors"]["violet"][6],
                variant="subtle",
                style={'width':'50px','minWidth':'40px','color':'rgb(40, 129, 205)'}
            ),
        ]
    ),
    
    # Processing
    dmc.Center(
        children = [
            dmc.ActionIcon(
                dmc.Tooltip(
                        label = "Processing",
                        position="left",
                        offset=3,
                        transitionProps={
                            "duration": 300,
                            "easing": "ease-in-out",
                            "transition":"scale-x"
                        },
                        radius ='10px',
                        color=dmc.DEFAULT_THEME["colors"]["violet"][6],
                        children = [DashIconify(icon="tabler:geometry", width=20, color=dmc.DEFAULT_THEME["colors"]["violet"][6]),]
                    ),
                id="btnProcessing",
                size="xl",
                color=dmc.DEFAULT_THEME["colors"]["violet"][6],
                variant="subtle",
                style={'width':'50px','minWidth':'40px'}
            ),
        ]
    ),
    # Clustering
    dmc.Center(
        children = [
            dmc.ActionIcon(
                dmc.Tooltip(
                        label = "Clustering",
                        position="left",
                        offset=3,
                        transitionProps={
                            "duration": 300,
                            "easing": "ease-in-out",
                            "transition":"scale-x"
                        },
                        radius ='10px',
                        color=dmc.DEFAULT_THEME["colors"]["violet"][6],
                        children = [DashIconify(icon="carbon:floorplan", width=20, color=dmc.DEFAULT_THEME["colors"]["violet"][6])]
                ),
                id="btnClustering",
                size="xl",
                color=dmc.DEFAULT_THEME["colors"]["violet"][6],
                variant="subtle",
                style={'width':'50px','minWidth':'40px'},
            ),
        ]
    )
    
]

style_navBar = {
    'borderRadius':'10px',
    "width": "4.5rem",
    "height": "fit-content",
    "boxSizing": "borderBox",
    "display": "flex",
    "flexDirection": "column",
    "backgroundColor": "rgb(255, 255, 255)",
    "borderRight": "0.0625rem solid rgb(233, 236, 239)",
    "padding": "1rem",
    'boxShadow': '10px 10px 10px 5px grey'
}


# filtered_list = filter(lambda x: x != "login", dash.page_registry.values())
app.layout = dmc.MantineProvider(
    children=[
        dmc.AppShell(
            children=[
                Header,
                dmc.AppShellNavbar(
                    id="navbar",
                    children=[
                        
                        dmc.Center(dmc.Group(dmc.Image(src=get_relative_path("/assets/favicon.png"),h=40, mb=20))), 
                        dmc.Divider(variant="solid", size="2px", mt=-11, mb=10),
                        dmc.Stack(
                            children = content_navbar,
                            # gap=
                        ),
                    ],
                    pl=10,
                    pr=10,
                    pt=10
                ),
                dmc.AppShellMain(
                    children = [
                        dcc.Location(id="url_app"),
                        dash.page_container
                    ],
                    style = {'paddingBottom':'180px'}
                ),
                Footer,
                
            ],
            header={"height": 60, "zIndex":1000},
            padding="md",
            layout="alt",
            navbar={
                "width": 80,
                "breakpoint": "sm",
                "collapsed": {"desktop": False, "mobile": True},
            },
            id="appshell",
        )
    ],
)

VALID_CREDENTIALS = ast.literal_eval(os.getenv('VALID_CREDENTIALS', ''))

# ================================================================================
#                           HEADER CALLBACK 
# ================================================================================
@callback(
    Output("appshell", "navbar"),
    Input("mobile-burger", "opened"),
    Input("desktop-burger", "opened"),
    State("appshell", "navbar"),
)
def toggle_navbar(mobile_opened, desktop_opened, navbar):
    navbar["collapsed"] = {
        "mobile": not mobile_opened,
        "desktop": not desktop_opened,
    }
    return navbar

clientside_callback(
    """ 
    (switchOn) => {
       document.documentElement.setAttribute('data-mantine-color-scheme', switchOn ? 'dark' : 'light');  
       return window.dash_clientside.no_update
    }
    """,
    Output("color-scheme-toggle", "id"),
    Input("color-scheme-toggle", "checked"),
)
# ================================================================================
#                           LOGIN CALLBACK 
# ================================================================================

@callback(
    [Output("login-message", "children"), Output("redirect", "href")],
    Input("login_btn", "n_clicks"),
    [State("login_username", "value"), State("login_password", "value")],
    prevent_initial_call=True
)
def verify_login(n_clicks, username, password):
    if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
        session['authenticated'] = True  # Store authentication in session
        return "", get_relative_path("/")  # Redirect to dashboard
    return "Invalid credentials. Try again.", dash.no_update

# Logout callback
@callback(
    Output("redirect-logout", "href"),
    Input("btn_logout", "n_clicks"),
    prevent_initial_call=True
)
def logout(n_clicks):
    session.pop('authenticated', None)  # Clear session
    return get_relative_path("/login")


if __name__ == '__main__':
    app.run_server(debug=False, port=8087, dev_tools_hot_reload=True)