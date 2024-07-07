import pandas as pd

from constants import categorical
import os

def read_data(filename):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url:

        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
        input_path = os.getenv('INPUT_FILE_PATTERN')
        df = pd.read_parquet(f'{filename}', storage_options=options)

    else:
        #input_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
        df = pd.read_parquet(filename)
    return df

def prepare_data(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def save_data(df_result, output_path):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
        df_result.to_parquet(output_path,
                      engine='pyarrow',
                      compression=None,
                      index=False, storage_options=options)
    else:
        df_result.to_parquet(output_path,
                      engine='pyarrow',
                      compression=None,
                      index=False)
    print("DF size", df_result.size)
    print(f"DataFrame saved to {output_path}")

