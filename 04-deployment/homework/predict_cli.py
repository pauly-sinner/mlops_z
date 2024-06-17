import pickle
import pandas as pd
import click
import numpy as np

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']
year = '2023'
month = '03'
output_file = 'predictions.parquet'

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def transform(df, dv):

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return X_val

def make_prediction(X_val, model):
    y_pred = model.predict(X_val)
    return y_pred

def output_preparing(df, y_pred, output_file):
    df_result = df.copy()
    df_result['ride_id'] = f'{year}/{month}_' + df_result.index.astype('str')
    df_result['prediction'] = y_pred

    df_result[['ride_id', 'prediction']].to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

@click.command()
@click.option('--year', required=True, type=str, help='Year parameter')
@click.option('--month', required=True, type=str, help='Month parameter')
def run(year, month):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')
    X_val = transform(df, dv)
    y_pred = make_prediction(X_val, model)
    print(np.mean(y_pred))
    output_preparing(df, y_pred, output_file)


if __name__ == "__main__":
    run()

