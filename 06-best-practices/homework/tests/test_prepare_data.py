import pandas as pd
from datetime import datetime

from pandas import Timestamp
from loading import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df = prepare_data(df)

    # Используем itertuples для извлечения данных
    extracted_data = list(df.itertuples(index=False, name=None))
    assert extracted_data ==  [('-1', '-1', Timestamp('2023-01-01 01:01:00'),Timestamp('2023-01-01 01:10:00'), 9.0),
                               ('1', '1', Timestamp('2023-01-01 01:02:00'), Timestamp('2023-01-01 01:10:00'), 8.0)]