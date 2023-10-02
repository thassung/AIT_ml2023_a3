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
    output = V3.calculate_price_class(29, 2023, 0, 1800, 130, submit)
    assert output == [2], f'Test case 1 failed'

def test_predict_car_price_case_2():
    output = V3.calculate_price_class(4, 2018, 0, 1500, 90, submit)
    assert output == [0], f'Test case 2 failed'

def test_predict_car_price_case_3():
    output = V3.calculate_price_class(None, None, None, None, None, submit)
    assert output == [0], f'Test case 3 failed'

def test_predict_car_price_case_4():
    output = V3.calculate_price_class(3, 2023, None, None, None, submit)
    assert output == [1], f'Test case 4 failed'

def test_predict_car_price_shape():
    output = V3.calculate_price_class(3, 2023, None, None, None, submit)
    assert output.shape == (1,), f'Expecting the shape to be (1,) but got {output.shape=}'

def test_predict_car_price_case_2():
    output = V3.calculate_selling_price_a3(3, 2023, None, None, None, submit)
    assert isinstance(output, str), f"Expecting output to be str but got {type(output)=}"