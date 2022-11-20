"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

import pandas as pd

def max_date(data: pd.DataFrame) -> pd.DataFrame:
    return {'max_date': data['Date'].max()}

def avg_high(data: pd.DataFrame) -> pd.DataFrame:
    return {'avg_high': data['High'].mean()}