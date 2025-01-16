from typing import Any, Dict
import pandas as pd
import random

def data_processing(general_df: pd.DataFrame, specific_df: pd.DataFrame) -> Dict[str, Any]:
    # Seleccionar 10 observaciones de cada lista para el conjunto de evaluación

    general_questions = general_df['question'].tolist()
    specific_questions = specific_df['question'].tolist()
    
    random.seed(42)
    eval_general_questions = random.sample(general_questions, 10)
    eval_specific_questions = random.sample(specific_questions, 10)

    # Crear las etiquetas correspondientes
    eval_questions = eval_general_questions + eval_specific_questions
    eval_labels = [0] * len(eval_general_questions) + [1] * len(eval_specific_questions)

    # Remover las preguntas de evaluación del conjunto de entrenamiento
    train_general_questions = [q for q in general_questions if q not in eval_general_questions]
    train_specific_questions = [q for q in specific_questions if q not in eval_specific_questions]

    train_questions = train_general_questions + train_specific_questions
    train_labels = [0] * len(train_general_questions) + [1] * len(train_specific_questions)
    
    return {'train_questions': train_questions, 'train_labels': train_labels, 'eval_questions': eval_questions, 'eval_labels': eval_labels}