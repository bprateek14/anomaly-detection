import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import IsolationForest


# def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
#     """Splits data into features and targets training and test sets.

#     Args:
#         data: Data containing features and target.
#         parameters: Parameters defined in parameters/data_science.yml.
#     Returns:
#         Split data.
#     """
#     X = data[parameters["features"]]
#     y = data["price"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
#     )
#     return X_train, X_test, y_train, y_test


def train_model(model_input_table: pd.DataFrame) -> IsolationForest:
    """Trains the linear regression model.

    Args:
        model_input_table: Training data for anomaly detection.
    Returns:
        Trained model.
    """
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(model_input_table)
    return model


def evaluate_model(
    model: IsolationForest, model_input_table: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame: 

    """Calculates and logs the coefficient of determination.

    Args:
        IsolationForest: Trained model.
        model_input_data: For predictions.
        processed_log_data: assigning predicted values to the logs.
    """
    anomaly_predictions = model.predict(model_input_table)
    logs['Anomaly'] = anomaly_predictions
    anomaly_count = len(logs.query("Anomaly==-1"))
    logger = logging.getLogger(__name__)
    logger.info("Model has detected %.3f anomalies.", anomaly_count)
    return logs
