from transformers import XLNetTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from modules.questions_dataset import QuestionsDataset
from typing import Any, Dict


class QuestionsDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_len):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "question_text": question,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea un DataLoader para el entrenamiento y otro para la evaluación a partir de los datos proporcionados.

    Args:
        data (Dict[str, Any]): Diccionario con los datos de entrenamiento y evaluación.

    Returns:
        DataLoader: DataLoader para el entrenamiento.
        DataLoader: DataLoader para la evaluación.
    """

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    # Reutilizar la clase QuestionsDataset y el tokenizer
    train_dataset = QuestionsDataset(
        data["train_questions"], data["train_labels"], tokenizer, max_len=32
    )
    eval_dataset = QuestionsDataset(
        data["eval_questions"], data["eval_labels"], tokenizer, max_len=32
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

    return {"train_dataloader": train_dataloader, "eval_dataloader": eval_dataloader}
