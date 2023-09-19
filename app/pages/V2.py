import dash
import joblib
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle


dash.register_page(
    __name__,
    path='/v2',
    title='ml 2023 a2 app',
    name='Car Seling Price Prediction App V2 (Linear Regression)'
)

# Create elements for app layout
f_1 = html.Div(
    [
        dbc.Label("Car brand: ", html_for="example-email"),
        dcc.Dropdown(id='f_1',
            options=[
                {'label': 'Ambassador', 'value': 1},
                {'label': 'Ashok', 'value': 2},
                {'label': 'Audi', 'value': 3},
                {'label': 'BMW', 'value': 4},
                {'label': 'Chevrolet', 'value': 5},
                {'label': 'Daewoo', 'value': 6},
                {'label': 'Datsun', 'value': 7},
                {'label': 'Fiat', 'value': 8},
                {'label': 'Force', 'value': 9},
                {'label': 'Ford', 'value': 10},
                {'label': 'Honda', 'value': 11},
                {'label': 'Hyundai', 'value': 12},
                {'label': 'Isuzu', 'value': 13},
                {'label': 'Jaguar', 'value': 14},
                {'label': 'Jeep', 'value': 15},
                {'label': 'Kia', 'value': 16},
                {'label': 'Land', 'value': 17},
                {'label': 'Lexus', 'value': 18},
                {'label': 'MG', 'value': 19},
                {'label': 'Mahindra', 'value': 20},
                {'label': 'Maruti', 'value': 21},
                {'label': 'Mercedes-Benz', 'value': 22},
                {'label': 'Mitsubishi', 'value': 23},
                {'label': 'Nissan', 'value': 24},
                {'label': 'Opel', 'value': 25},
                {'label': 'Peugeot', 'value': 26},
                {'label': 'Renault', 'value': 27},
                {'label': 'Skoda', 'value': 28},
                {'label': 'Tata', 'value': 29},
                {'label': 'Toyota', 'value': 30},
                {'label': 'Volkswagen', 'value': 31},
                {'label': 'Volvo', 'value': 32},
                {'label': 'Other/Unknown', 'value': 0}
            ], value = None,
            placeholder=' select car brand'),
        dbc.FormText(
            " \n",
            color="secondary",
        ),
    ],
    style={"width": "15%"},
    className="mb-3",
)

f_2 = html.Div(
    [
        dbc.Label("Built year: ", html_for="example-email"),
        dbc.Input(id="f_2", type="number", value = None, placeholder=" ex. 1999, 2015"),
        dbc.FormText(
            "",
            color="secondary",
        ),
    ],
    className="mb-3",
)

f_3 = html.Div(
    [
        dbc.Label("Transmission: ", html_for="example-email"),
        dcc.Dropdown(id='f_3',
            options=[
                {'label': 'Automatic', 'value': 0},
                {'label': 'Manual', 'value': 1}
            ], 
            value = None,
            placeholder=' select transmission type'),
        dbc.FormText(
            " \n",
            color="secondary",
        ),
    ],
    style={"width": "15%"},
    className="mb-3",
)

f_4 = html.Div(
    [
        dbc.Label("Engine capacity: ", html_for="example-email"),
        dbc.Input(id="f_4", type="number", value = None, placeholder=" unit is CC"),
        dbc.FormText(
            " CC",
            color="secondary",
        ),
    ],
    className="mb-3",
)

f_5 = html.Div(
    [
        dbc.Label("Max power: ", html_for="example-email"),
        dbc.Input(id="f_5", type="number", value = None, placeholder=" unit is bhp"),
        dbc.FormText(
            " bhp",
            color="secondary",
        ),
    ],
    className="mb-3",
)

submit_button_f = html.Div([
            dbc.Button(id="submit_button_f", children="Submit", color="primary", className="me-1"),
            dbc.Label("  "),
            html.Output(id="selling_price_f", 
                        children='')
            ], style={'marginTop':'10px'})


form =  dbc.Form([
            f_1, f_2, f_3, f_4, f_5,
            submit_button_f,
            
        ],
        className="mb-3")


# Explain Text
text = html.Div([
    html.H1("Car Predicing (predict pricing) V2"),
    html.P("The model is a LinearRegression model."),
])

# Dataset Example
from dash import Dash, dash_table
# import pandas as pd
# df = pd.read_csv('./code/data/Cars - Cars.csv')

# table = dbc.Table.from_dataframe(df.head(50), 
#                         striped=True, 
#                         bordered=True, 
#                         hover=True,
#                         responsive=True,
#                         size='sm'
#                             )

intro = dbc.Container([html.Div("This is a Car Selling Price Prediction Web App. Users can input 5 values including:", style={"margin-left": "15px"}),
                                           html.Div("- Car brand: A dropdown selection of car brand", style={"margin-left": "55px"}),
                                           html.Div("- Built year: Built year of the car", style={"margin-left": "55px"}),
                                           html.Div("- Transmission: A gear transmission type of cars", style={"margin-left": "55px"}),
                                           html.Div("- Engine capacity: The size of the engine (unit is CC)", style={"margin-left": "55px"}),
                                           html.Div("- Max power: Maximum force produced by car's engine (unit is bhp)", style={"margin-left": "55px"}),
                                        ], fluid=True)

layout =  dbc.Container([
        intro,
        text,
        form,
    ], fluid=True)
    
@callback(
    Output(component_id="selling_price_f", component_property="children"),
    State(component_id="f_1", component_property="value"),
    State(component_id="f_2", component_property="value"),
    State(component_id="f_3", component_property="value"),
    State(component_id="f_4", component_property="value"),
    State(component_id="f_5", component_property="value"),
    Input(component_id='submit_button_f', component_property='n_clicks'),
    prevent_initial_call=True
)

def calculate_selling_price_f(x_1, x_2, x_3, x_4, x_5, submit):
    
    theta = np.array([11.8483747322865, 5.11556371e-01,  5.00582796e-01,  1.52814768e-01,  4.80810114e-01,
        1.45044074e-01,  3.68618533e-03,  3.04312161e-01,  8.62618975e-01,
        2.89742600e-01,  3.65952722e-03,  2.23932358e-01,  1.79973965e-01,
        5.32000931e-08,  5.62629479e-01,  6.79000414e-01,  6.54094464e-01,
        3.02687769e-03,  7.14580349e-01,  1.37401355e-02,  1.36780542e-08,
        6.67855661e-02,  3.21519964e-01, -2.02666706e-02,  5.80049091e-01,
        8.31820660e-01,  5.49341243e-01,  6.19639082e-02,  4.40936905e-01,
       -4.15636238e-03, -5.12503458e-02,  5.85848914e-01,  5.74092033e-01,
        2.70964181e-01,  8.13208243e-01,  5.42766988e-01,  8.15381002e-01])
    scaler = pickle.load(open('./pages/codeV2/model/scaler.pkl','rb'))
    ## scale engine and max_power
    if x_1 is None:
        x_1 = 0
    if x_2 is None:
        x_2 = 2015.0
    if x_3 is None:
        x_3 = 1
    if x_4 is None:
        x_4 = 1248.0
    if x_5 is None:
        x_5 = 82.85
    print(x_1,x_2,x_3,x_4,x_5)
    tbs = pd.DataFrame({'year': [x_2], 'engine':[x_4], 'max_power':[x_5]})
    tbs = scaler.transform(tbs)
    x_2, x_4, x_5 = tbs[0][0], tbs[0][1], tbs[0][2]

    ## create dummies value for brand
    brand_list = ['Ambassador','Ashok','Audi','BMW','Chevrolet','Daewoo','Datsun','Fiat',
                    'Force','Ford','Honda','Hyundai','Isuzu','Jaguar','Jeep',
                    'Kia','Land','Lexus','MG','Mahindra','Maruti','Mercedes-Benz',
                    'Mitsubishi','Nissan','Opel','Peugeot','Renault','Skoda','Tata',
                    'Toyota','Volkswagen','Volvo']

    col_order = ['year','transmission','engine','max_power','b_Ambassador','b_Ashok','b_Audi',
                'b_BMW','b_Chevrolet','b_Daewoo','b_Datsun','b_Fiat','b_Force','b_Ford',
                'b_Honda','b_Hyundai','b_Isuzu','b_Jaguar','b_Jeep','b_Kia','b_Land',
                'b_Lexus','b_MG','b_Mahindra','b_Maruti','b_Mercedes-Benz','b_Mitsubishi',
                'b_Nissan','b_Opel','b_Peugeot','b_Renault','b_Skoda','b_Tata',
                'b_Toyota','b_Volkswagen','b_Volvo']

    b_cols = np.zeros(len(brand_list))
    if x_1 > 0:
        b_cols[x_1] = 1
    
    X = np.array([1, x_2, x_3, x_4, x_5])
    X = np.concatenate([X, b_cols])
    pred = X @ theta
    return f"Predicted car price is: {np.exp(pred):.2f}"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)

# if __name__ == "__main__":
#     app.run_server(debug=True)