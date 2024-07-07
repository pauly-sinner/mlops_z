year = "example_2023"
month ="example_03"
input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
output_file = f'taxi_type=yellow_year={year}_month={month}.parquet'
categorical = ['PULocationID', 'DOLocationID']