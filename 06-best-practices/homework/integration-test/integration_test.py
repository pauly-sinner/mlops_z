import os
import pandas as pd
from datetime import datetime

# Set the environment variables for testing
#os.environ['INPUT_FILE_PATTERN'] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
#os.environ['OUTPUT_FILE_PATTERN'] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
os.environ['S3_ENDPOINT_URL'] = "http://localhost:4566"


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

# Define the output path for January 2023
default_input_pattern = f's3://nyc-duration/in/{"2023"}-{"01"}.parquet'
input_path = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)

# Check if S3_ENDPOINT_URL is set
s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
if s3_endpoint_url:
    options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }
    df.to_parquet(input_path,
    engine='pyarrow',
    compression=None,
    index=False, storage_options=options)
else:
    df.to_parquet(input_path,
    engine='pyarrow',
    compression=None,
    index=False)
print("DF size", df.size)
print(f"DataFrame saved to {input_path}")