import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)
logging.info('Loading Python libraries ...')

import yaml
import os
import sys
import contextlib
import io
import pandas as pd
import numpy as np

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def download(**kwargs): 

    download_folder = kwargs['download_folder']
    download_file = kwargs['download_file']
    kaggle_dataset = kwargs['kaggle_dataset']
    kaggle_file = kwargs['kaggle_file']

    column_impacting_the_split = kwargs['column_impacting_the_split']
    current_subset_size = kwargs['current_subset_size']
    prob_coeff = kwargs['prob_coeff']
    time_period_column_name = kwargs['time_period_column_name']

    number_of_nan_values = kwargs['number_of_nan_values']

    # The full folder path to dlownload the file to
    download_path = os.path.join(curr_dir, os.pardir, download_folder)

    # Check whether the file has already been downloaded
    if os.path.exists(os.path.join(download_path, download_file)):

        logging.info("The CSV file has already been downloaded.")
    
    else:

        # Code may raise an exception ...
        try: 
            
            from dotenv import load_dotenv
            load_dotenv()

            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            with contextlib.redirect_stdout(io.StringIO()): # Logging suppression
                # Download the file
                api.dataset_download_files(dataset = kaggle_dataset, path = download_path, quiet=True, unzip=True)
                # Rename the file
                os.rename(os.path.join(download_path, kaggle_file), os.path.join(download_path, download_file))

            logging.info("The CSV file was successfully downloaded from Kaggle.")

        # ... if the kaggle.json file is nowhere to be found
        except OSError as e:
            # Handle the exception
            logging.error("The Kaggle credentials are missing!")
            sys.exit(1)

        # Any other error
        except Exception:
            logging.error("Unknown error. The file wasn't downloaded.")
            sys.exit(1)

    # Reload the file
    Data = pd.read_csv(os.path.join(curr_dir, os.pardir, download_folder, download_file))
    # Split the file into the 'reference and 'current' parts. This will be used by EvidentlyAI later on
    dataset_size = Data.shape[0]
    # Probabilities used to split the data set to the 'reference and the 'current' subsets
    # For prob_coeff=0 the probabilities to fall into either dataset are the same
    probabilities = np.exp(prob_coeff * ((np.arange(dataset_size, dtype=float) / (dataset_size - 1))))
    # Normalization
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(Data.sort_values(column_impacting_the_split, ascending=True).index, 
                                       size=int(current_subset_size * dataset_size), p=probabilities, replace=False)
    # Add the split column values
    Data[time_period_column_name] = pd.Series(Data.index.isin(sampled_indices)).map({True: 'current', False: 'reference'})
    logging.info("Dataset split into the current and the current subsets, {} and {} in size respectively.".format(dataset_size, sampled_indices.size))

    # Add missing values
    # Get random indices to replace with NaN
    nan_indices = np.random.choice(Data.shape[0], size=number_of_nan_values, replace=False)
    Data.loc[nan_indices, [column_impacting_the_split]] = np.nan

    # Save the modified data
    try:
        Data.to_csv(os.path.join(curr_dir, os.pardir, download_folder, download_file), index=False)
        logging.info("The modified Kaggle CSV data was successfully saved.")
    except:
        logging.error("The modified Kaggle CSV data was was not saved!")
        sys.exit(1)

# The download.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['download']

if __name__ == "__main__":
    download(**params)