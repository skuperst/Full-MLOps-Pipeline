
from pathlib import Path
import os, yaml
import logging

from utils.update_check_utils import update_check

from datetime import datetime
import sys

# Initiate logging
logging.basicConfig(level=logging.INFO)

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

# The full path to the file with the MLFlow experiment name
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))
mlflow_experiment_name_file_path =  params['mlflow_experiment_name_file_path']

# Check whether the current file exists
if update_check():
    logging.info('Preparing MLFLOW experiment name.')
    # Save the updated MLFlow experiment name
    try:
        with open(mlflow_experiment_name_file_path, "w") as file:
            exp_name = "Model Re-Run: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)"))
            yaml.dump({'exp_name': exp_name}, file) # For example, "Model Re-Run: 2024/12/24 (13:42)"
        logging.info("The experiment name was assigned the name \'{}\'".format(exp_name))
    except:
        logging.error("The experiment name was not assigned!")
        sys.exit(1)
else:
    logging.info('No current data downloaded. Skipping this stage with empty input data.')
    # Create empty outs file
    Path(mlflow_experiment_name_file_path).touch()
    data_folder = params['preprocess']['data_folder']
    input_file = params['preprocess']['input_file']
    Path(os.path.join(curr_dir, os.pardir, data_folder, input_file)).touch()
    
  