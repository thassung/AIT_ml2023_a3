from dash import Dash, html, Output, Input, dcc
import dash_bootstrap_components as dbc
import dash


app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Model ver.1", href="/v1")),
        dbc.NavItem(dbc.NavLink("Model ver.2", href="/v2")),
        dbc.NavItem(dbc.NavLink("Model ver.3", href="/v3")),
    ],
    brand="ML2023 Car Selling Price Prediction A2",
    brand_href="/",
    color="primary",
    dark=True,
)


dash.register_page("Homepage", 
                   layout = dbc.Container([html.H1("Car Selling Price Prediction Apps"),
                                           html.Div("This is a Car Selling Price Prediction Web App. Users can input 5 values including:", style={"margin-left": "15px"}),
                                           html.Div("- Car brand: A dropdown selection of car brand", style={"margin-left": "55px"}),
                                           html.Div("- Built year: Built year of the car", style={"margin-left": "55px"}),
                                           html.Div("- Transmission: A gear transmission type of cars", style={"margin-left": "55px"}),
                                           html.Div("- Engine capacity: The size of the engine (unit is CC)", style={"margin-left": "55px"}),
                                           html.Div("- Max power: Maximum force produced by car's engine (unit is bhp)", style={"margin-left": "55px"}),
                                           html.Div("There are two verions of the app—V1 and V2.", style={"margin-left": "35px"}),
                                           html.Div("V1 uses Random Forest model and V2 uses Linear Regression model", style={"margin-left": "15px"}),
                                           html.Div("The V2 model gives nondecreasing price increase upon increasing in max_power and engine which can make more sense when comparing two car models", style={"margin-left": "15px"}),
                                           html.Br(),
                                           html.Div("\nPlease navigate with the top navigation bar >>>>>", style={"margin-left": "15px"})
                                        ], fluid=True),
                    path="/", order=0)

app.layout = html.Div([
    navbar,
    dash.page_container
])

# app.layout = html.Div([
#     html.H1('Car Selling Price Prediction Apps'),
#     html.Div([
#         html.Div(
#             dcc.Link(f"{page['relative_path']}", href=page["relative_path"])
#         ) for page in dash.page_registry.values()
#     ]),
#     dash.page_container
# ])

