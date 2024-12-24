import logging
import os
from datetime import datetime
from ruamel.yaml import YAML
import sys

# Initiate logging
logging.basicConfig(level=logging.INFO)

# Initialize YAML object
yaml = YAML()
yaml.preserve_quotes = True  # Preserves quotes, useful for strings

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

# The full path to the file with the MLFlow experiment name
mlflow_experiment_name_file_path =  yaml.load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['mlflow_experiment_name_file_path']

# Save the updated MLFlow experiment name
try:
    with open(mlflow_experiment_name_file_path, "w") as file:
        exp_name = "Experiment: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)"))
        yaml.dump({'exp_name': exp_name}, file) # For example, "Experiment: 2024/12/24 (13:42)"
    logging.info("The experiment name was assigned the name \'{}\'".format(exp_name))
except:
    logging.error("The experiment name was not assigned!")
    sys.exit(1)