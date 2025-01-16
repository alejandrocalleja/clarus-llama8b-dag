"""
This module defines the data_processing function used by the pipeline orchestrator to perform data preprocessing. 
This function defines the logic for data preprocessing. Any adidtional function needed to perform this step can 
be defined within this script itself or split into different scripts and included in the Process directory.
"""

from sklearn.model_selection import train_test_split
from typing import Dict, Any
import pandas as pd

def data_processing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data preprocessing on the input dataframe and transform it into train and validation datasets.

    In this code example, the input dataframe is divided into train_x, train_y, val_x and val_y dataframes.

    Args:
        df: The input dataframe containing the data to be preprocessed.

    Return:
        A dictionary containing the preprocessed data.
    """

    # Train and validation data split
    train, val = train_test_split(df,random_state = 0,shuffle = False)

    # Extract features and target variables for train and validation data
    train_x = train.drop(['quality'], axis=1)
    val_x = val.drop(['quality'], axis=1)
    train_y = train[['quality']]
    val_y = val[['quality']]

    return {'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y}