"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import max_date
from .nodes import avg_high


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=max_date,
            inputs='doge',
            outputs='max-date'
        ),
        node(
            func=avg_high,
            inputs='doge',
            outputs='avg-high'
        )
    ])
