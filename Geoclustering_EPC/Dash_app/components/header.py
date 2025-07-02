import  dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import dcc, get_relative_path, html


def get_icon(icon, rotate_):
    return DashIconify(icon=icon, height=18, rotate = rotate_, color = dmc.DEFAULT_THEME["colors"]["violet"][6])


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


Header = dmc.AppShellHeader(
    dmc.Group(
        [
            dmc.Flex(
                children = [
                    
                    dmc.Anchor(
                        children = [get_icon(icon="gravity-ui:geo-pin",rotate_=0),dmc.Text("EPC - Analysis", style={"fontSize": 14}, fw=500, ml=2)],
                        # id="anchor_",
                        className = "anchor_1",
                        href=get_relative_path("/"),
                    ),
                    dmc.Anchor(
                        children = [get_icon(icon="carbon:text-link-analysis",rotate_=0),dmc.Text("Synthetic EPC", style={"fontSize": 14}, fw=500, ml=2)], 
                        # id="anchor_",
                        className = "anchor_1",
                        href=get_relative_path("/synthetic_epc"),
                    ),
                ],
                h="100%",
                align="center",
            ),    
            dmc.Title("Energy Performance Certificate - Geoclustering", size="xl", fw=600, c="violet", className="gradient-text"),
            dmc.Group(
                children = [
                    dmc.Text("Powered by", size="xs", fw=600, c="violet", className="gradient-text", mb=25),
                    dcc.Link(
                                children = dmc.Avatar(
                                    src="assets/mostly_ai_logo.png", radius=0,
                                        style ={
                                            'width':'100%',
                                            'height': '25px',
                                            # 'mar':'3px'
                                        }
                                    ),  
                                href="https://mostly.ai/",
                            ),
                    
                    theme_toggle,
                ],
                gap="md"
            )
        ],
        ml=0,
        justify="space-between",
        style={"flex": 1},
        h="100%",
        px="md",
    )
)





# Header = dmc.AppShellHeader(
#     dmc.Group(
#                 [
#                     dcc.Location(id="redirect-logout", refresh=True),
#                     dmc.Group(
#                         [
#                             dmc.Burger(
#                                 id="burger",
#                                 size="sm",
#                                 hiddenFrom="sm",
#                                 opened=False,
#                             ),
#                             dmc.Image(src=get_relative_path("/assets/moderate_logo.png"),h=40),
#                         ]
#                     ),
#                     dmc.Group(
#                         children=[
#                             dmc.Group(
#                                 children = [
#                                     dmc.NavLink(
#                                         label="Processing",
#                                         href=get_relative_path("/processing"),
#                                         id={"type": "navlink", "index": get_relative_path("/processing")},
#                                         variant="light",
#                                         color="violet",
#                                         style = {'borderRadius': '10px'}
#                                     ),
#                                     dmc.NavLink(
#                                         label="Clustering",
#                                         href=get_relative_path("/clustering"),
#                                         id={"type": "navlink", "index": get_relative_path("/clustering")},
#                                         variant="light",
#                                         color="violet",
#                                         style = {'borderRadius': '10px'}
#                                     ),
#                                     theme_toggle,
#                                     dmc.Text("Hi, Moderate", size="xl", fw=600),
#                                     dmc.Button(
#                                             'Logout', 
#                                             id="btn_logout", 
#                                             leftSection = DashIconify(icon="solar:logout-broken", color="white", width=25),
#                                             radius="md", color="gray"
#                                     ) 
#                                 ]
#                             )
#                         ],
#                         ml="xl",
#                         gap=0,
#                         visibleFrom="sm",
#                     ),
#                 ],
#                 justify="space-between",
#                 style={"flex": 1},
#                 h="100%",
#                 px="md",
#             ),
    # dmc.Group(
    #     [
    #         dcc.Location(id="redirect-logout", refresh=True),
    #         dmc.Group(
    #             [
    #                 dmc.Burger(
    #                     id="burger",
    #                     size="sm",
    #                     hiddenFrom="sm",
    #                     opened=False,
    #                 ),
    #                 dmc.Image(src=get_relative_path("/assets/moderate_logo.png"),h=40),
    #             ]
    #         ),
    #         dmc.Group(
    #             children = [
    #                 theme_toggle,
    #                 dmc.Text("Hi, Moderate", size="xl", fw=600),
    #                 dmc.Button(
    #                     'Logout', 
    #                     id="btn_logout", 
    #                     leftSection = DashIconify(icon="solar:logout-broken", color="white", width=25),
    #                     radius="md", color="gray"
    #                 ) 
    #             ]
    #         )
        # ],
    #     justify="space-between",
    #     style={"flex": 1},
    #     h="100%",
    #     px="md",
    # ),
# )


Header_home = dmc.AppShellHeader(
    h=60,
    id="header",
    children = [
        dcc.Location(id="redirect-logout", refresh=True),
        dmc.Group(
            id="group_header",
            children= [
                dcc.Link(
                    children = [
                        dmc.Avatar(
                            src=get_relative_path("/assets/moderate_logo.png"), radius=0,
                            style ={
                                'width':'100%',
                                'height': '32px',
                            }
                        )
                    ],
                    href = get_relative_path("/"),
                ),
                dmc.Group(
                    children = [
                        dmc.Text("Hi, Cordia", size="xl", fw=600),
                        dmc.Button(
                            'Logout', 
                            id="btn_logout", 
                            leftSection = DashIconify(icon="solar:logout-broken", color="white", width=25),
                            radius="md", color="gray") 
                    ]
                )
            ],
            justify="space-between"
        )
    ]
)