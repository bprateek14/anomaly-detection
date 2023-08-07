import pandas as pd
import numpy as np
import json
import logging




def combine_data(log_1:pd.DataFrame, log_2:pd.DataFrame) -> pd.DataFrame:
    """Combine log files
    
    Args:
        logs_1: Raw data.
        logs_2: Raw data.
    Returns:
        Combined log files.
    """
    logs = pd.concat([log_1,log_2],axis=0)
    return logs


def preprocess_logs(combined_logs: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        logs: Raw data.
    Returns:
        Preprocessed data, with `clean_logs`.
    """
    combined_logs['DOMAIN_DATA'] = combined_logs['DOMAIN_DATA'].apply(lambda x: x.replace('\n',''))
    combined_logs['DOMAIN_DATA'] = combined_logs['DOMAIN_DATA'].apply(lambda x: x.replace(' ',''))
    return combined_logs


def parsed_logs(preprocessed_logs: pd.DataFrame) -> pd.DataFrame:
    parsed_logs = []

    for log_string in preprocessed_logs['DOMAIN_DATA'].to_list():
        try:
            log_dict = json.loads(log_string)
            parsed_logs.append(log_dict)
        except json.JSONDecodeError:
            print("Error parsing log:", log_string)
    
    # Convert parsed logs into a DataFrame
    preprocessed_logs = pd.DataFrame(parsed_logs)
    return preprocessed_logs


def create_model_input_table(parsed_logs: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical features into numerical using one-hot encoding.

    Args:
        log_df: Preprocessed data for shuttles.
    
    Returns:
        Model input table.

    """
    parsed_logs = parsed_logs.drop(['stats','stack'],axis=1).replace(np.nan,0).replace(' ',0)
    # Select features for the model
    features = parsed_logs.columns

    # Convert categorical features into numerical using one-hot encoding
    encoded_log_df = pd.get_dummies(parsed_logs[features])
    return encoded_log_df
