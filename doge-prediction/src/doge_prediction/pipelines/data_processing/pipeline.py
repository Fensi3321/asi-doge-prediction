"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import hello_world


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=hello_world,
            inputs='doge',
            outputs='max-date'
        )
    ])
