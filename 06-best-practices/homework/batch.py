#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import os
from model import load_model
from loading import read_data, prepare_data, save_data
from constants import input_file, output_file, categorical
os.environ['S3_ENDPOINT_URL'] = "http://localhost:4566"
year = sys.argv[1]
month = sys.argv[2]
def get_input_path(year, month):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url:
        default_input_pattern = f's3://nyc-duration/in/{year}-{month}.parquet'
        input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
        return input_pattern.format(year=year, month=month)
    else:
        return f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'

def get_output_path(year, month):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url:
        default_output_pattern = f's3://nyc-duration/out/{year}-{month}.parquet'
        output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
        return output_pattern.format(year=year, month=month)
    else:
        return f'taxi_type=yellow_year={year}_month={month}.parquet'

def get_s3_url(year, month):
    default_output_pattern = f's3://nyc-duration/'
    output_pattern = os.getenv('S3_ENDPOINT_URL', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year :str, month: str):

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    df = read_data(input_file)
    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    df = prepare_data(df)

    dicts = df[categorical].to_dict(orient='records')

    dv, lr = load_model('model.bin')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    print('Sum of predicted durations', sum(df_result['predicted_duration']) )
    save_data(df_result, output_file)

if __name__ == "__main__":
    main(year, month)
