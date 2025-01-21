import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

logging.info('Loading Python libraries ...')

import os
import sys
import yaml
import pickle

import json

import pandas as pd


logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def intergration(**kwargs):

    # for key, value in kwargs.items():
    #     globals()[key] = value

    reference_data_file_path = kwargs['reference_data_file_path']
    current_data_file_path = kwargs['current_data_file_path']

    try:

        # Load CURRENT and REFERENCE datasets
        reference_data = pd.read_csv(reference_data_file_path) 
        current_data = pd.read_csv(current_data_file_path)

        pd.concat([reference_data, reference_data], ignore_index=True).to_csv(reference_data_file_path)

        logging.info('Concatenated reference with current data.')

    except:
        logging.error("Failed to concatenate reference with current data.!")
        sys.exit(1)
        
# The intergration.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['evidentlyai']

if __name__ == "__main__":
    intergration(**params)