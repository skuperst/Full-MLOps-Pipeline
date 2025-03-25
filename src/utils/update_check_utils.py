import os
import yaml
import logging

def update_check():
    
    # Initiate logging
    logging.basicConfig(level=logging.INFO)

    # Current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # The download_current_data.py parameters from the params.yaml file
    params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, os.pardir, "params.yaml")))['download_current_data']
    
    data_folder = params['data_folder']
    raw_current_data_file = params['raw_current_data_file']
   
    file_exists = os.path.exists(os.path.join(curr_dir, os.pardir, os.pardir, data_folder, raw_current_data_file))
    if file_exists:
        return (os.stat(os.path.join(curr_dir, os.pardir, os.pardir, data_folder, raw_current_data_file)).st_size > 0)
    else:
        return False



