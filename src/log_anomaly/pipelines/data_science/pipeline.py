from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs="model_input_table",
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "model_input_table", "logs"],
                outputs="final_prediction",
                name="prediction_node",
            ),
        ]
    )
