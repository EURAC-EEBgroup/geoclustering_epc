from dash import callback, Output, Input, ALL, ctx, State  

@callback(
    Output({"type": "navlink_", "index": ALL}, "active"), 
    Input("_pages_location", "pathname"), 
    State({"type": "navlink_", "index": ALL}, "active"), 
)
def update_navlinks(pathname, activated):
    return [control["id"]["index"] == pathname for control in ctx.outputs_list]
    # if pathname == "/epc_clustering/piemonte/processing":
    #     return [True, False]
    # elif pathname == "/epc_clustering/piemonte/clustering":
    #     return [False, True]
    # else:
    #     return [True, False]
    # print(pathname)
    # print(activated)
    # print(activate)
    # return activate