"""
This module defines the `select_best_model` function used by the pipeline orchestrator to select the best model 
from an MLflow experiment based on a specified metric.

This function connects to the MLflow tracking server and retrieves the best model based on the specified metric 
from the MLflow experiment. The metric can be either maximized or minimized based on the provided metric type. 

Any additional functions or utilities required for this step can be defined within this script itself or split 
into different scripts and included in the Process directory.
"""

import mlflow
from mlflow.tracking.client import MlflowClient
import config

def select_best_model():
    """
    Select the best model based on a specified metric from the MLflow experiment.

    This function connects to the MLflow tracking server using the provided endpoint. It retrieves the best model
    based on the specified metric from the MLflow experiment. The metric can be either maximized or minimized based
    on the provided metric type. 

    Returns:
        None
    """

    endpoint = config.MLFLOW_ENDPOINT
    experiment = config.MLFLOW_EXPERIMENT
    metric = config.METRIC_BM
    metric_type = config.METRIC_BM_TYPE

    client = MlflowClient(endpoint)
    mlflow.set_tracking_uri(endpoint)
    mlflow.set_experiment(experiment)  

    # Retrieve the best model based on the specified metric
    if metric_type=='max':
        runs = client.search_runs(experiment_ids=[client.get_experiment_by_name(experiment).experiment_id],
                                order_by=[f"metrics.{metric} DESC"],max_results=1)
    else:
        runs = client.search_runs(experiment_ids=[client.get_experiment_by_name(experiment).experiment_id],
                                order_by=[f"metrics.{metric} ASC"],max_results=1) 
    
    
    # Do not change
    # --------------------------------------------------------------------------------
    # Get the best run id
    best_run_id = runs[0].info.run_id
    # Get the run details
    run = mlflow.get_run(best_run_id)
    # Accessing the Minio path to the artifact
    artifact_path = run.info.artifact_uri + '/model'
    # Accessing the model metrics
    model_metrics = run.data.metrics
    print(f'BEST RUN: {best_run_id}')

    return {'best_run': best_run_id, 'artifact_path': artifact_path, 'model_metrics': model_metrics}

