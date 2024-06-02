from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#from mlops.utils.data_preparation.encoders import vectorize_features
#from mlops.utils.data_preparation.feature_selector import select_features


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
def compute_duration(data):
    data = data.copy()
    data['duration'] = ((data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']) 
                          / np.timedelta64(1, 'm'))
    return data

def drop_outliters(data):
    data = data[(data['duration']>=1) &
                                 (data['duration'] <=60)]
    return data

def one_hot(data, cols):
    data = data[cols].astype(str).copy()
    dicts = data.to_dict(orient='records')
    return dicts
categorical = ['PULocationID', 'DOLocationID']


@data_exporter
def export(
    data: DataFrame,
) -> Tuple[
    LinearRegression,
    DictVectorizer
]:
    target = 'duration'
    #target = kwargs.get('target', 'duration')


    df_march = compute_duration(data)

    data_cut_outliers = drop_outliters(df_march)

    train_dicts = one_hot(data_cut_outliers, categorical)

    dv = DictVectorizer()
    matrix_train = dv.fit_transform(train_dicts)
    y_train = data_cut_outliers[target].values

    lr = LinearRegression()
    lr.fit(matrix_train, y_train)

    y_pred = lr.predict(matrix_train)

    print(f"Intercept: {lr.intercept_}")

    return lr, dv


