import pandas as pd

def read_general_data() -> pd.DataFrame:
    general_questions_df = pd.read_csv("src/llm_classificator/Data/general_questions.csv")
    return general_questions_df