import os
import yaml
from dotenv import load_dotenv
import mlflow
import logging
from datetime import datetime

def configure_mlflow():

    # Initiate logging
    logging.basicConfig(level=logging.INFO)

    # Current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # The full path to the file with the experiment current name
    experiment_name_file_path = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, os.pardir, "params.yaml")))['mlflow_experiment_name_file_path']

    # The experiment name from the params.yaml file
    experiment_name = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, os.pardir, experiment_name_file_path)))['exp_name']
    # experiment_name = 'Model Created: 2025/03/22'

    # Load the .env file
    load_dotenv(override=True)
    # Set environment variables for authentication
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER_NAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PASSWORD")

    try:
        # Set tracking URI programmatically (reads from environment variable)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        # .. add set the experiment
        mlflow.set_experiment(experiment_name)
        logging.info("The MLFlow experiment tracking was successfully set.")
    except:
        logging.error("The MLFlow tracking failed!")
