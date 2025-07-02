from dash import callback, Input, Output

@callback(
    Output("login_username", "persistence"),
    Output("login_username", "persistence_type"),
    Output("login_password", "persistence"),
    Output("login_password", "persistence_type"),
    Input("remember_me","checked")
)
def remember_username_password(checked):
    if checked:
        return True, "session", True, "session"
    return False, "local", False, "local"

