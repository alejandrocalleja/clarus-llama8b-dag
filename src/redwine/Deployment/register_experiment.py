"""
This module provides the register_experiment function, which is utilized by the pipeline orchestrator (Airflow) to 
register the experiment into IDS TrueConnector. If any additional auxiliary functions are required to accomplish 
this step, they can be defined within the same script or separated into different scripts and included in the Data
directory.
"""

import json
import config
import psycopg2
from psycopg2 import Error
from IDS_templates.register_experiment_main import handle_post
from IDS_templates.rest_ids_consumer_connector import RestIDSConsumerConnector

def register_experiment_rds(res: dict = None):
    """
        This function implements the logic to regsiter a new experiment into a rds database
    """
    # Connect to your PostgreSQL database
    try:
        connection = psycopg2.connect(
            user=config.POSTGRES_USERNAME,
            password=config.POSTGRES_PASSWORD,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DATABASE
        )

        cursor = connection.cursor()

        # SQL query to insert data into best_model_tracking table
        postgres_insert_query = """ 
            INSERT INTO best_model_tracking (experiment_id, run_id, minio_path, metrics, datetime) 
            VALUES (%s, %s, %s, %s, current_timestamp)
        """

        # Values to be inserted into the table
        best_run_id = res['best_run']
        artifact_path = res['artifact_path']
        model_metrics = res['model_metrics']
        print(f"res: {res}")
        print(f"best_run_id: {best_run_id}")
        print(f"artifact_path: {artifact_path}")
        print(f"model_metrics: {model_metrics}")
        record_to_insert = (config.MLFLOW_EXPERIMENT, best_run_id, artifact_path, json.dumps(model_metrics))


        # Execute the SQL command
        cursor.execute(postgres_insert_query, record_to_insert)

        # Commit changes to the database
        connection.commit()
        print("Record inserted successfully into best_model_tracking table")
        
         # Close database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    except (Exception, Error) as error:
        print("Error while working with PostgreSQL", error)
        raise error


def register_experiment(res: dict = None):
    """
        This function implements the logic to regsiter a new experiment into a database and into IDS consumer tureconnector
    """
    register_experiment_rds(res)

    ids_consumer = RestIDSConsumerConnector()
    print("Register experiment task:")

    if ids_consumer.is_artifact_internal_registered_by_resource_title(config.MLFLOW_EXPERIMENT ,config.TRUE_CONNECTOR_CLOUD_IP)==False:
        return handle_post(config.MLFLOW_EXPERIMENT,'Primera prueba',config.TRUE_CONNECTOR_CLOUD_IP)
    else:
        print('The experiment is already registered')
    