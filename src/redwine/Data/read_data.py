"""
This module provides the read_data function, which is utilized by the pipeline orchestrator (Airflow) for data ingestion. 
The function implements the logic to ingest the data and transform it into a pandas format. If any additional auxiliary 
functions are required to accomplish this step, they can be defined within the same script or separated into different 
scripts and included in the Data directory.
"""

import pandas as pd

def read_data() -> pd.DataFrame:
    """
    The function implements the logic to ingest the data and transform it into a pandas format.

    In this code example, a csv file is retrieved from a url.

    Return:
        A Pandas DataFrame representing the content of the specified file.
    """

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ';')
    
    print(df)
    return df