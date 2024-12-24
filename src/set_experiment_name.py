import logging
import os
from datetime import datetime
from ruamel.yaml import YAML

# Initialize YAML object
yaml = YAML()
yaml.preserve_quotes = True  # Preserves quotes, useful for strings

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
# The params.yaml file full path
params_file_path = os.path.join(curr_dir, os.pardir, 'mlflow_experiment_name.yaml')

# Save the updated configuration back to the params.yaml file
with open(params_file_path, "w") as file:
   print(datetime.now().strftime("%Y/%m/%d (%H:%M)"))
   yaml.dump({'exp_name': "Experiment: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)"))}, file)

logging.info("The experiment date in params.yaml was updated successfully.")