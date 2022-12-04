"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import max_date
from .nodes import avg_high
from .nodes import prepare_data
from .nodes import train_model
from .nodes import train_model2


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
            outputs='avg_high'
        ),
        node(
            func=prepare_data,
            inputs='doge',
            outputs='doge_processed'
        ),
        node(
            func=train_model,
            inputs='doge_processed',
            outputs=None
        ),
        node(
            func=train_model2,
            inputs='doge_processed',
            outputs=None
        )
    ])
