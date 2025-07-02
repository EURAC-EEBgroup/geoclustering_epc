import dash 
import dash_mantine_components as dmc
from dash import html, dcc, get_relative_path
from dash_iconify import DashIconify
from components.footer import Footer
from flask import session

dash.register_page(__name__,path=f"/login")

# Simulated User Database

# ===============================================================
#                   BOX COMPONENTS LOGIN 
# ===============================================================
boxLogin = dmc.Card(
    children = [
        dcc.Location(id="redirect", refresh=True),
        dmc.CardSection(
            children=[
                dmc.Center(dmc.Title(f"Sign in", order=2)),
            ],
            withBorder=False,
            inheritPadding=True,
            py="md",
        ),
        dmc.CardSection(
            children = [
                dmc.Box(
                    children=[
                        dmc.Stack(
                            pos="relative",
                            p=5,
                            w=300,
                            children=[
                                dmc.TextInput(
                                    # label="Username",
                                    placeholder="Your username",
                                    leftSection=DashIconify(icon="radix-icons:person"),
                                    id="login_username",
                                    mt=10,
                                    # persistence=True,
                                    # persistence_type="session"
                                ),
                                dmc.TextInput(
                                    # label="Password",
                                    placeholder="Your password",
                                    leftSection=DashIconify(icon="radix-icons:lock-closed"),
                                    id="login_password",
                                    mt=10,
                                    # persistence=True,
                                    # persistence_type="session"
                                ),
                                dmc.Switch(
                                    size="sm",
                                    radius="lg",
                                    label="Remember me",
                                    checked=True,
                                    mt=10,
                                    id="remember_me"
                                ),
                                dmc.Button(
                                    "Login", id="login_btn", variant="outline", fullWidth=True, radius="md", color="violet"
                                ),
                                dmc.Divider(label="or"),
                                dmc.Button(
                                    "Sign up", id="signup_btn", variant="outline", fullWidth=True, radius="md", color="violet", disabled=True
                                ),
                                html.Div(id="login-message", style={"color": "red", "margin-top": "10px"}),
                                html.Div(children="", id="login-output"),
                            ],
                        ),
                    ]
                )
            ],
            inheritPadding=True,
            py="md"
        )
    ],
    withBorder=True,
    shadow="md",
    radius="lg",
)

# ===============================================================
#                   FOOTER
# ===============================================================



# ===============================================================
#                   MAIN LOGIN 
# ===============================================================

main_login = html.Div(
    children = [
        dmc.Container(
            id="container_login",
            children = [
                dmc.Center(dmc.Title("Welcome", order=1, c="white", mt=80)),
                dmc.Center(
                    boxLogin,
                ),
            ]
        ),
        Footer
    ],
    style = {'height':'100vh'}
    
)





mainContainer = dmc.BackgroundImage(
    src=get_relative_path("/assets/clustering_images.png"),
    children =  [
        html.Div(
            main_login
        )
    ]
)

def layout():
    if session.get('authenticated'):
        return dcc.Location(href=get_relative_path("/"), id="redirect-home")
    return mainContainer