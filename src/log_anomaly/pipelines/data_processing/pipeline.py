from kedro.pipeline import Pipeline, node, pipeline

from .nodes import combine_data, preprocess_logs, parsed_logs,create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=combine_data,
                inputs=["log_1","log_2"],
                outputs="logs",
                name="combine_data_node",
            ),
            node(
                func=preprocess_logs,
                inputs="logs",
                outputs="preprocessed_logs",
                name="preprocess_log_node",
            ),
             node(
                func=parsed_logs,
                inputs="preprocessed_logs",
                outputs="parsed_logs",
                name="parsed_log_node",
            ),
            node(
                func=create_model_input_table,
                inputs="parsed_logs",
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
