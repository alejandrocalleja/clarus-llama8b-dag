"""
This module defines the `elasticNet_model_training` function used by the pipeline orchestrator to train a machine 
learning model using ElasticNet regularization. This function defines the logic for training the model and evaluating 
its performance.

Any additional functions or utilities required for this step can be defined within this script itself or split into 
different scripts and included in the Process directory.
"""

from typing import Dict, Any
from sklearn.linear_model import ElasticNet
from datetime import datetime
from Models import utils

def elasticNet_model_training(data: Dict[str, Any]):
    """
    Train an ElasticNet model using the provided data.

    The function trains an ElasticNet model on the given training data and evaluates its performance on the validation data.

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
    
    estimator_name = "ElasticNet"
    alpha_list = [0.5, 0.6]
    l1_ratio_list = [0.5]
    # runs_list = []

    for alpha in alpha_list:
        for l1_ratio in l1_ratio_list:
            run_name = f"ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})"
            hyperparams = {'alpha': alpha, 'l1_ratio': l1_ratio}

            # Model training
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            # Model training performance evaluation
            predicted_qualities = lr.predict(train_x)
            training_metrics = utils.eval_metrics(train_y, predicted_qualities, 'train')
            print(f"training_metrics: {training_metrics}")

            # Model test
            predicted_qualities = lr.predict(val_x)
            validation_metrics = utils.eval_metrics(val_y, predicted_qualities, 'validation')

            # Track the run
            last_run = utils.track_run(run_name, estimator_name, hyperparams, training_metrics, validation_metrics, lr)
            # runs_list.append(last_run)

    # return {'elasticNet_runs_list': runs_list}