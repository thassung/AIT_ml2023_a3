import dash
import joblib
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle


dash.register_page(
    __name__,
    path='/v1',
    title='ml 2023 a1 app',
    name='Car Seling Price Prediction App V1 (Random Forest)'
)

# Create elements for app layout
x_1 = html.Div(
    [
        dbc.Label("Car brand: ", html_for="example-email"),
        dcc.Dropdown(id='x_1',
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

x_2 = html.Div(
    [
        dbc.Label("Built year: ", html_for="example-email"),
        dbc.Input(id="x_2", type="number", value = None, placeholder=" ex. 1999, 2015"),
        dbc.FormText(
            "",
            color="secondary",
        ),
    ],
    className="mb-3",
)

x_3 = html.Div(
    [
        dbc.Label("Transmission: ", html_for="example-email"),
        dcc.Dropdown(id='x_3',
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

x_4 = html.Div(
    [
        dbc.Label("Engine capacity: ", html_for="example-email"),
        dbc.Input(id="x_4", type="number", value = None, placeholder=" unit is CC"),
        dbc.FormText(
            " CC",
            color="secondary",
        ),
    ],
    className="mb-3",
)

x_5 = html.Div(
    [
        dbc.Label("Max power: ", html_for="example-email"),
        dbc.Input(id="x_5", type="number", value = None, placeholder=" unit is bhp"),
        dbc.FormText(
            " bhp",
            color="secondary",
        ),
    ],
    className="mb-3",
)

submit_button = html.Div([
            dbc.Button(id="submit_button", children="Submit", color="primary", className="me-1"),
            dbc.Label("  "),
            html.Output(id="selling_price", 
                        children='')
            ], style={'marginTop':'10px'})


form =  dbc.Form([
            x_1, x_2, x_3, x_4, x_5,
            submit_button,
            
        ],
        className="mb-3")


# Explain Text
text = html.Div([
    html.H1("Car Predicing (predict pricing) V1"),
    html.P("The model is a RandomForest model."),
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
    Output(component_id="selling_price", component_property="children"),
    State(component_id="x_1", component_property="value"),
    State(component_id="x_2", component_property="value"),
    State(component_id="x_3", component_property="value"),
    State(component_id="x_4", component_property="value"),
    State(component_id="x_5", component_property="value"),
    Input(component_id='submit_button', component_property='n_clicks'),
    prevent_initial_call=True
)

def calculate_selling_price(x_1, x_2, x_3, x_4, x_5, submit):
    
    model = joblib.load('./pages/codeV1/model/rf_random_selling_price.model')
    scaler = pickle.load(open('./pages/codeV1/model/scaler.pkl','rb'))
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
    tbs = pd.DataFrame({'engine':[x_4], 'max_power':[x_5]})
    tbs = scaler.transform(tbs)
    x_4, x_5 = tbs[0][0], tbs[0][1]

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
        b_cols[x_1-1] = 1
    
    X = np.array([x_2, x_3, x_4, x_5])
    X = np.concatenate([X, b_cols])
    X = pd.DataFrame([X], columns =col_order)
    pred = np.exp(model.predict(X))[0]
    return f"Predicted car price is: {pred:.2f}"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)

# if __name__ == "__main__":
#     app.run_server(debug=True)