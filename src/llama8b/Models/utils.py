import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

from tqdm import tqdm
import torch
from urllib.parse import urlparse
import time
import config
import pandas as pd
import sys

sys.path.insert(1, "/git/clarus-llm-dag/src/llm_classificator")
from Models.custom_pyfunc import MPT

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "question"),
    ]
)
output_schema = Schema([ColSpec(DataType.string, "candidate")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame({"question": ["What is machine learning?"]})


def eval_metrics(actual, pred, mode="test"):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {f"{mode}_rmse": rmse, f"{mode}_mae": mae, f"{mode}_r2": r2}


def eval_model(model, dataloader, device):
    model = model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    eval_loss = total_loss / len(dataloader)
    eval_accuracy = correct_predictions.double() / len(dataloader.dataset)
    return eval_loss, eval_accuracy


def track_run(
    run_name: str,
    model_name: str,
    estimator_name: str,
    hyperparams: dict,
    training_metrics: dict,
    validation_metrics: dict,
    model: any,
):

    # Auxiliar functions and connection stablishment
    client = MlflowClient(config.MLFLOW_ENDPOINT)
    mlflow.set_tracking_uri(config.MLFLOW_ENDPOINT)
    try:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    except:
        time.sleep(10)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    warnings.filterwarnings("ignore")

    # # Model registry
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    # if tracking_url_type_store != "file":
    #     mlflow.sklearn.log_model(model, "model", registered_model_name=run_name)
    # else:
    #     mlflow.sklearn.log_model(model, "model")

    mlflow.start_run(run_name=run_name, tags={"estimator_name": estimator_name})

    active_run = mlflow.active_run()
    # track hypreparameters
    for key, value in hyperparams.items():
        mlflow.log_param(key, value)

    # Track training metrics
    for key, value in training_metrics.items():
        mlflow.log_metric(key, value)

    # Track validation metrics
    for key, value in validation_metrics.items():
        mlflow.log_metric(key, value)

    # Model registry
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store != "file":
        mlflow.pyfunc.log_model(
            model_name,
            python_model=MPT(),
            artifacts={},
            input_example=input_example,
            signature=signature,
            registered_model_name=run_name,
        )
    else:
        mlflow.pyfunc.log_model(
            model_name,
            python_model=MPT(),
            artifacts={},
            input_example=input_example,
            signature=signature,
        )

    mlflow.end_run()

    # Print report
    tr_keys = list(training_metrics.keys())
    tst_keys = list(validation_metrics.keys())

    print(f"{run_name}:")
    print("  TRAIN:")
    print("     LOSS: %s" % training_metrics[tr_keys[0]])
    print("     ACCURACY: %s" % training_metrics[tr_keys[1]])
    print("  VALIDATION:")
    print("     LOSS: %s" % validation_metrics[tst_keys[0]])
    print("     ACCURACY: %s" % validation_metrics[tst_keys[1]])
