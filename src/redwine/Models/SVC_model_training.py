"""
This module defines the `svc_model_training` function used by the pipeline orchestrator to train a machine 
learning model using ElasticNet support vector machine. This function defines the logic for training the model and evaluating 
its performance.

Any additional functions or utilities required for this step can be defined within this script itself or split into 
different scripts and included in the Process directory.
"""

from typing import Dict, Any
from datetime import datetime
from sklearn.svm import SVC
from Models import utils

def svc_model_training(data: Dict[str, Any]):
    """
    Train an SVM model using the provided data.

    The function trains an SVC model on the given training data and evaluates its performance on the validation data.

    Args:
        data: A dictionary containing the preprocessed data.

    Returns:
        None
    """

    train_x = data['train_x']
    train_y = data['train_y']
    val_x = data['val_x']
    val_y = data['val_y']
    
    print('Execution init datetime: ' + str(datetime.now()))
    
    estimator_name = "SVC"
    kernel_list = ['sigmoid']
    # runs_list = []

    for kernel in kernel_list:
        run_name = f"SVC(kernel={kernel})"
        hyperparams = {'kernel': kernel}

        # Model training
        lr = SVC(kernel = kernel, random_state=0)
        lr.fit(train_x, train_y)

        # Model training performance evaluation
        predicted_qualities = lr.predict(train_x)
        training_metrics = utils.eval_metrics(train_y, predicted_qualities, 'train')

        # Model test
        predicted_qualities = lr.predict(val_x)
        validation_metrics = utils.eval_metrics(val_y, predicted_qualities, 'validation')

        # Track the run
        last_run = utils.track_run(run_name, estimator_name, hyperparams, training_metrics, validation_metrics, lr)
        # runs_list.append(last_run)

    # return {'SVC_runs_list': runs_list}