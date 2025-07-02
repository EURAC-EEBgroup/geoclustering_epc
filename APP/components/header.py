import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import html, dcc, get_relative_path 


HeaderAdmin = dmc.Header(
    className="z-index-90",  
    height=70,
    style = {'paddingRight':'0','paddingLeft':'0',
            'color':'white'
                },
    fixed = True, 
    p="md",
    children=[
        dmc.Container(
            fluid=True,
            children=dmc.Group(
                position="apart",
                align="flex-start",
                children=[
                    html.Div(
                        children = [
                            dmc.Text(
                                id = "UserName",
                                variant="gradient",
                                gradient={"from": "blue", "to": "green", "deg": 45},
                                style={"fontSize": 30}
                            )
                        ],
                        style = {'backgroundColor':'transparent', 'color':'black','paddingLeft': '0px'}
                    ),
                    dmc.Group(
                        position="right",
                        align="center",
                        spacing="xl",
                        children=[
                            dmc.Menu(
                                [
                                    dmc.MenuTarget(
                                        dmc.ActionIcon(
                                            DashIconify(
                                                icon="ep:menu",
                                                width=20,
                                                color= 'black',
                                                )
                                            ),
                                        ),
                                    dmc.MenuDropdown(
                                        [
                                            html.A(
                                                dmc.MenuItem(
                                                    "Account",
                                                    icon=DashIconify(icon="ri:admin-fill"),
                                                ),
                                                id="btn_admin_to_user",
                                                target = "_blank",
                                                href="/admin",
                                            ),
                                            html.A(
                                                dmc.MenuItem(
                                                    "Logout",
                                                    icon=DashIconify(icon="icon-park-outline:logout"),
                                                ),
                                                id="btn_logout_admin",
                                                href="/logout",
                                                style = {'MarginRight':'20px'}
                                            )
                                        ]
                                    ),
                                ]
                            ),
                            dmc.Button("", 
                                id="rightDrawer",
                                leftIcon=[DashIconify(icon="akar-icons:three-line-horizontal", width=20)],
                                style = {'backgroundColor':'transparent', 'color':'#63B3ED','paddingLeft': '0px'}
                            ),
                        ],
                    ),                            
                ],
            ),
        ),        
    ],
)