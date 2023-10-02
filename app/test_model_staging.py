from utils import load_mlflow
import numpy as np
import pandas as pd
import pytest

stage = "Staging"
def test_load_model():
    model = load_mlflow(stage=stage)
    assert model

@pytest.mark.depends(on=['test_load_model'])
def test_model_input():
    model = load_mlflow(stage=stage)
    X = np.array([[1.        , 2.26912613, 0.        , 0.68273025, 1.09344826,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 1.        , 0.        ,
        0.        , 0.        ]])
    phat = model.predict(X) # type:ignore
    assert phat

@pytest.mark.depends(on=['test_model_input'])
def test_model_output():
    model = load_mlflow(stage=stage)
    X = np.array([[1.        , 2.26912613, 0.        , 0.68273025, 1.09344826,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 1.        , 0.        ,
        0.        , 0.        ]])
    pred = model.predict(X) # type:ignore
    assert pred.shape == (1,), f"{pred.shape=}"

@pytest.mark.depends(on=['test_load_model'])
def test_model_coeff():
    model = load_mlflow(stage=stage)
    assert model.W.shape == (37,4), f"{model.coef_.shape=}" # type:ignore