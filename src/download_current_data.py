import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)
logging.info('Loading Python libraries ...')

import yaml, sys
import os, csv, io
from google.cloud import storage
from dotenv import load_dotenv 
from pathlib import Path

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def download_current_data(**kwargs):

    try:
        raw_current_data_file = kwargs['raw_current_data_file']
        data_folder = kwargs['data_folder']

        # Load environment variables
        load_dotenv()
        bucket_name = os.getenv('BUCKET_NAME')
        # Get the credentials path from the environment variable
        credentials_path = os.path.join(curr_dir, os.pardir, os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

        # Create a Cloud Storage client using service account credentials.
        client = storage.Client.from_service_account_json(credentials_path)
        # # Access the bucket
        bucket = client.get_bucket(bucket_name)

        # File path to save the raw current data
        file_path = os.path.join(curr_dir, os.pardir, data_folder, raw_current_data_file)

        # Check whether there is a new (current) file called raw_current_data_file ('raw_current_data.csv')
        if raw_current_data_file in [b.name for b in bucket.list_blobs()]:
            csv.writer(open(file_path, 'w', newline='', encoding='utf-8')).writerows(
                csv.reader(io.StringIO(bucket.blob(raw_current_data_file).download_as_text())))
            logging.info("Raw current data was successfully downloaded from the Google Storage bucket.")
        else:
            # If none, create an empty file
            Path(file_path).touch()
            logging.info("No current data in the Google Storage bucket. Created an empty data file.")
        
    except:
        logging.error("Failed to download raw current data from Google bucket!")
        sys.exit(1)

# The download_current_data.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['download_current_data']

if __name__ == "__main__":
    download_current_data(**params)    