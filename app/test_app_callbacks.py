from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import pytest
import app
from pages import V3

import pandas as pd
import numpy as np
import pickle
import mlflow

submit = 1
def test_predict_car_price_case_1():
    output = V3.calculate_selling_price_a3(29, 2023, 0, 1800, 130, submit)
    assert output == 'Predicted car selling price is between 3614999.5 INR and 5407499.75 INR (class 2)'

def test_predict_car_price_case_2():
    output = V3.calculate_selling_price_a3(4, 2018, 0, 1500, 90, submit)
    assert output == 'Predicted car selling price is less than 1822499.25 INR (class 0)'

def test_predict_car_price_case_3():
    output = V3.calculate_selling_price_a3(None, None, None, None, None, submit)
    assert output == 'Predicted car selling price is less than 1822499.25 INR (class 0)'

def test_predict_car_price_case_4():
    output = V3.calculate_selling_price_a3(3, 2023, None, None, None, submit)
    assert output == 'Predicted car selling price is between 1822499.25 INR and 3614999.5 INR (class 1)'

def test_predict_car_price_case_2():
    output = V3.calculate_selling_price_a3(3, 2023, None, None, None, submit)
    assert isinstance(output, str), f"Expecting output to be str but got {type(output)}"