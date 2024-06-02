import os
import pickle
import click
import mlflow
from typing import Tuple
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


EXPERIMENT_NAME = "linear-regression-mage"
#TRACKING_URI = "http://127.0.0.1:5000"
TRACKING_URI = "http://mlflow_server:5000"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def log_model(logging_data : Tuple[
    LinearRegression,
    DictVectorizer]):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    model, dv = logging_data[0], logging_data[1]
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear-regression")
        mlflow.log_params(model.get_params())
        with open("dv.p", "wb") as f:
            pickle.dump(dv, f)
    
        mlflow.log_artifact('dv.p')
