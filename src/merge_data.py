import logging, os, yaml
from pathlib import Path
# Initiate logging
logging.basicConfig(level=logging.INFO)

from utils.update_check_utils import update_check

update_check = update_check()
if update_check:

    # If there is an update load the heavy Python libraries 
    logging.info('Loading Python libraries ...')

    import sys
    import pandas as pd

    logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def merge_data(**kwargs):

    preprocessed_reference_data_file = kwargs['preprocessed_reference_data_file']
    preprocessed_current_data_file = kwargs['preprocessed_current_data_file']
    data_folder = kwargs['data_folder']
    merged_data_file = kwargs['merged_data_file']

    if update_check:
        logging.info('Merging current and reference data.')
    else:
        logging.info('No current data downloaded. Skipping merging with empty outputs.')
        # Create empty outs file
        Path(os.path.join(curr_dir, os.pardir, data_folder, merged_data_file)).touch()
        exit()

    try:
        # Load CURRENT and REFERENCE datasets
        reference_data = pd.read_csv(os.path.join(curr_dir, os.pardir, data_folder, preprocessed_reference_data_file)) 
        current_data = pd.read_csv(os.path.join(curr_dir, os.pardir, data_folder, preprocessed_current_data_file)) 

        pd.concat([reference_data, current_data], ignore_index=True).to_csv(os.path.join(curr_dir, os.pardir, data_folder, merged_data_file), index=False)

        logging.info('Concatenated reference with current data.')

    except:
        logging.error("Failed to concatenate reference with current data!")
        sys.exit(1)

# The preprocess.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['merge_data']

if __name__ == "__main__":
    merge_data(**params)
