import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)
logging.info('Loading Python libraries ...')

import yaml
import os
import sys
import contextlib
import io

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def download(**kwargs): #kaggle_dataset, kaggle_file, download_folder, download_file

    # The full folder path to dlownload the file to
    download_path = os.path.join(curr_dir, os.pardir, kwargs['download_folder'])

    # Check whether the file has already been downloaded
    if os.path.exists(os.path.join(download_path, kwargs['download_file'])):

        logging.info("The CSV file has already been downloaded.")
    
    else:

        # Code may raise an exception ...
        try:  

            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            with contextlib.redirect_stdout(io.StringIO()): # Logging suppression
                # Download the file
                api.dataset_download_files(dataset = kwargs['kaggle_dataset'], path = download_path, quiet=True, unzip=True)
                # Rename the file
                os.rename(os.path.join(download_path, kwargs['kaggle_file']), os.path.join(download_path, kwargs['download_file']))

            logging.info("The CSV file is successfully downloaded.")

        # ... if the kaggle.json file is nowhere to be found
        except OSError as e:
            # Handle the exception
            logging.error("The Kaggle credentials are missing!")
            sys.exit(1)

        # Any other error
        except Exception:
            logging.error("Unknown error. The file wasn't downloaded.")
            sys.exit(1)


# The download.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['download']

if __name__ == "__main__":
    download(**params)