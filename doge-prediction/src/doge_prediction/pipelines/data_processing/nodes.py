"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

import pandas as pd

def hello_world(data: pd.DataFrame) -> pd.DataFrame:
    return {'max_date': data['Date'].max()}