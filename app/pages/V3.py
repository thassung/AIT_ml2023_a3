import dash
import joblib
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import pandas as pd ##
import numpy as np  ##
import pickle       ##
import mlflow       ##


dash.register_page(
    __name__,
    path='/v3',
    title='ml 2023 a3 app',
    name='Car Seling Price Prediction App V3 (Logistic Regression)'
)

# Create elements for app layout
a3_1 = html.Div(
    [
        dbc.Label("Car brand: ", html_for="example-email"),
        dcc.Dropdown(id='a3_1',
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

a3_2 = html.Div(
    [
        dbc.Label("Built year: ", html_for="example-email"),
        dbc.Input(id="a3_2", type="number", value = None, placeholder=" ex. 1999, 2015"),
        dbc.FormText(
            "",
            color="secondary",
        ),
    ],
    className="mb-3",
)

a3_3 = html.Div(
    [
        dbc.Label("Transmission: ", html_for="example-email"),
        dcc.Dropdown(id='a3_3',
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

a3_4 = html.Div(
    [
        dbc.Label("Engine capacity: ", html_for="example-email"),
        dbc.Input(id="a3_4", type="number", value = None, placeholder=" unit is CC"),
        dbc.FormText(
            " CC",
            color="secondary",
        ),
    ],
    className="mb-3",
)

a3_5 = html.Div(
    [
        dbc.Label("Max power: ", html_for="example-email"),
        dbc.Input(id="a3_5", type="number", value = None, placeholder=" unit is bhp"),
        dbc.FormText(
            " bhp",
            color="secondary",
        ),
    ],
    className="mb-3",
)

submit_button_a3 = html.Div([
            dbc.Button(id="submit_button_a3", children="Submit", color="primary", className="me-1"),
            dbc.Label("  "),
            html.Output(id="selling_price_a3", 
                        children='')
            ], style={'marginTop':'10px'})


form =  dbc.Form([
            a3_1, a3_2, a3_3, a3_4, a3_5,
            submit_button_a3,
            
        ],
        className="mb-3")


# Explain Text
text = html.Div([
    html.H1("Car Predicing (predict pricing) V2"),
    html.P("The model is a LogisticRegression model."),
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
                                           html.Div("For this model, the prediction will be in range of price not the exact price", style={"margin-left": "15px"}),
                                        ], fluid=True)

layout =  dbc.Container([
        intro,
        text,
        form,
    ], fluid=True)

@callback(
    Output(component_id="selling_price_a3", component_property="children"),
    State(component_id="a3_1", component_property="value"),
    State(component_id="a3_2", component_property="value"),
    State(component_id="a3_3", component_property="value"),
    State(component_id="a3_4", component_property="value"),
    State(component_id="a3_5", component_property="value"),
    Input(component_id='submit_button_a3', component_property='n_clicks'),
    prevent_initial_call=True
)


    

def calculate_price_class(x_1, x_2, x_3, x_4, x_5):
    mlflow.set_tracking_uri('https://mlflow.cs.ait.ac.th/')
    model = mlflow.sklearn.load_model('models:/st124323-a3-model/staging')
    scaler = pickle.load(open('./codeV3/model/scaler.pkl','rb'))

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

    tbs = pd.DataFrame({'year': [x_2], 'engine':[x_4], 'max_power':[x_5]})
    tbs = scaler.transform(tbs)
    x_2, x_4, x_5 = tbs[0][0], tbs[0][1], tbs[0][2]

    ## dummies for brand 
    ## x_1 starts from 1
    ## 0 is Other/Unknown
    # brand_list = ['Ambassador','Ashok','Audi','BMW','Chevrolet','Daewoo','Datsun','Fiat',
    #                 'Force','Ford','Honda','Hyundai','Isuzu','Jaguar','Jeep',
    #                 'Kia','Land','Lexus','MG','Mahindra','Maruti','Mercedes-Benz',
    #                 'Mitsubishi','Nissan','Opel','Peugeot','Renault','Skoda','Tata',
    #                 'Toyota','Volkswagen','Volvo']

    col_order = ['year','transmission','engine','max_power','b_Ambassador','b_Ashok','b_Audi',
                'b_BMW','b_Chevrolet','b_Daewoo','b_Datsun','b_Fiat','b_Force','b_Ford',
                'b_Honda','b_Hyundai','b_Isuzu','b_Jaguar','b_Jeep','b_Kia','b_Land',
                'b_Lexus','b_MG','b_Mahindra','b_Maruti','b_Mercedes-Benz','b_Mitsubishi',
                'b_Nissan','b_Opel','b_Peugeot','b_Renault','b_Skoda','b_Tata',
                'b_Toyota','b_Volkswagen','b_Volvo']
    
    X = pd.DataFrame(0, index=np.array([1]), columns=col_order)
    X.iloc[:,:4] = x_2, x_3, x_4, x_5
    X.iloc[:,x_1+3] = 1
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    phat = model.predict(X)

    return phat

def calculate_selling_price_a3(x_1, x_2, x_3, x_4, x_5, submit):
    phat = calculate_price_class(x_1, x_2, x_3, x_4, x_5)[0]
    if phat == 0:
        return f"Predicted car selling price is less than 1822499.25 INR (class {phat})"
    elif phat == 1:
        return f"Predicted car selling price is between 1822499.25 INR and 3614999.5 INR (class {phat})"
    elif phat == 2:
        return f"Predicted car selling price is between 3614999.5 INR and 5407499.75 INR (class {phat})"
    elif phat == 3:
        return f"Predicted car selling price is more than 5407499.75 INR (class {phat})"
    
    return f'Uh oh. Sumting wong (phat: {phat[0]})'     

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)

# if __name__ == "__main__":
#     app.run_server(debug=True)