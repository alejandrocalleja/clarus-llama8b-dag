import pandas as pd

def read_data() -> pd.DataFrame:
    specific_questions_df = pd.read_csv("src/llm_classificator/Data/specific_questions.csv")
    return specific_questions_df