from transformers import XLNetForSequenceClassification, AdamW
from typing import Any, Dict
from datetime import datetime
from tqdm import tqdm
from torch import cuda, bfloat16
import torch
from Models import utils


def check_if_gpu_available():
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        # Get the current CUDA device
        device = torch.cuda.current_device()
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

        # Get GPU memory usage
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9} GB")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(device) / 1e9} GB")
        print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(device) / 1e9} GB")

    # If CUDA is not available, print a message
    else:
        print("CUDA is not available. Using CPU.")


def train_epoch(model, dataloader, optimizer, device):
    model = model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(dataloader)
    train_accuracy = correct_predictions.double() / len(dataloader.dataset)

    return train_loss, train_accuracy


def xlNet_model_training(data: Dict[str, Any], epochs=1):
    check_if_gpu_available()
    run_name = "XLNet_model"
    estimator_name = "XLNet"
    hyperparams = {"epochs": epochs, "optimizer": "AdamW"}
    training_metrics = {}
    validation_metrics = {}
    lr = 1e-5

    train_dataloader = data["train_dataloader"]
    eval_dataloader = data["eval_dataloader"]

    model_name = "xlnet-base-cased"
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    print("Execution init datetime: " + str(datetime.now()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}")

    eval_loss, eval_accuracy = utils.eval_model(model, eval_dataloader, device)

    training_metrics = {"train_loss": train_loss, "train_accuracy": train_accuracy}
    validation_metrics = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}

    utils.track_run(run_name, model_name, estimator_name, hyperparams, training_metrics, validation_metrics, model)

    return "training done"
