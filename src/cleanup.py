import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)
logging.info('Loading Python libraries ...')

import yaml
import os

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def cleanup(**kwargs):

    data_folder = kwargs['data_folder']
    raw_current_data_file = kwargs["raw_current_data_file"]
    merged_data_file = kwargs['merged_data_file']
    test_X_file = kwargs['test_X_file']
    test_y_file = kwargs['test_y_file']
    preprocessed_current_data_with_predictions_file = kwargs['preprocessed_current_data_with_predictions_file']
    preprocessed_current_data_file=kwargs['preprocessed_current_data_file']

    # Delete local current files at the end of pipeline
    for f in [raw_current_data_file, merged_data_file, 
              preprocessed_current_data_file, preprocessed_current_data_with_predictions_file, test_X_file, test_y_file]:
        f_path = os.path.join(curr_dir, os.pardir, data_folder, f)
        if os.path.exists(f_path):
            os.remove(f_path)
            logging.info("Deleted file {}.".format(f))
        else:
            logging.info("File {} was scheduled for deletion, but it does not exist.".format(f))

# The cleanup.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['cleanup']

if __name__ == "__main__":
    cleanup(**params)