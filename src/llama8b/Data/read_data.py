from typing import Tuple
import requests
import pandas as pd
from io import StringIO
import config


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    general_questions_url = "https://api-generator.retool.com/VAtSxo/general_questions"
    specific_questions_url = (
        "https://api-generator.retool.com/yJCzkf/specific_questions"
    )

    try:
        general_questions_response = requests.get(general_questions_url)
        specific_questions_response = requests.get(specific_questions_url)

        general_questions = general_questions_response.json()
        specific_questions = specific_questions_response.json()
        # Convert the JSON data to a DataFrame
        general_df = pd.DataFrame(general_questions)
        specific_df = pd.DataFrame(specific_questions)

        general_df = general_df.drop(columns=["id"])
        specific_df = specific_df.drop(columns=["id"])

        return general_df, specific_df
    except Exception as e:
        print(e)
        return None, None
