import os
import sys
import yaml
import logging
import contextlib
import io

logging.basicConfig(level=logging.INFO)

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
# The download.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, '..', "params.yaml")))['download']

def download(kaggle_dataset, kaggle_file, download_folder, download_file):

    # The full folder path to dlownload the file to
    download_path = os.path.join(curr_dir, '..', download_folder)

    # Check whether the file has already been downloaded
    if os.path.exists(os.path.join(download_path, download_file)):

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
                api.dataset_download_files(dataset = kaggle_dataset, path = download_path, quiet=True, unzip=True)
                # Rename the file
                os.rename(os.path.join(download_path, kaggle_file), os.path.join(download_path, download_file))

            logging.info("The CSV file is successfully downloaded.")

        # ... if the kaggle.json file is nowhere to be found
        except OSError as e:
            # Handle the exception
            logging.error(f"The Kaggle credentials are missing!")
            sys.exit(1)

        # Any other error
        except Exception:
            logging.error(f"Unknown error. The file wasn't downloaded.")
            sys.exit(1)

if __name__ == "__main__":
    download(kaggle_dataset = params["kaggle_dataset"], 
             kaggle_file = params["kaggle_file"], 
             download_folder = params["download_folder"],
             download_file = params["download_file"])