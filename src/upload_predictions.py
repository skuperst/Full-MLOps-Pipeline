import logging, os, yaml
# Initiate logging
logging.basicConfig(level=logging.INFO)

from utils.update_check_utils import update_check

update_check = update_check()
if update_check:

    # If there is an update load the heavy Python libraries 
    logging.info('Loading Python libraries ...')

    import sys
    from google.cloud import storage
    from dotenv import load_dotenv
    import pickle
    import pandas as pd 
    
    logging.info('Done!')
else:
    from pathlib import Path    

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def upload_predictions(**kwargs):

    preprocessed_current_data_file = kwargs['preprocessed_current_data_file']
    data_folder = kwargs['data_folder']
    model_file_path = kwargs['new_model_file_path']
    prediction_column = kwargs['prediction_column']
    preprocessed_current_data_with_predictions_file = kwargs["preprocessed_current_data_with_predictions_file"]

    if update_check:
        logging.info('Uploading the preprocessed current data file with model predictions to the bucket..')
    else:
        logging.info('No current data downloaded. Skipping upload.')
        # Create empty outs file
        Path(os.path.join(curr_dir, os.pardir, data_folder, preprocessed_current_data_with_predictions_file)).touch()     
        exit()
        
    # Load the train model
    model = pickle.load(open(model_file_path,'rb'))

    # Download preprocessed current data
    Data = pd.read_csv(os.path.join(curr_dir, os.pardir, data_folder, preprocessed_current_data_file))

    try:
        # Calculate the test set predictions
        Data['prediction'] = model.predict(Data.drop(prediction_column, axis=1))
        # Save the file
        Data.to_csv(os.path.join(curr_dir, os.pardir, data_folder, preprocessed_current_data_with_predictions_file))

        logging.info("Made current data prediction.")

    except:
        logging.error("Failed to make current data prediction!")
        sys.exit(1)

    # Upload
    try:
        # Load environment variables
        load_dotenv()
        bucket_name = os.getenv('BUCKET_NAME')
        # Get the credentials path from the environment variable
        credentials_path = os.path.join(curr_dir, os.pardir, os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

        # Create a Cloud Storage client using service account credentials.
        client = storage.Client.from_service_account_json(credentials_path)
        # # Access the bucket
        bucket = client.get_bucket(bucket_name)

        # Create a blob (file object in GCS)
        blob = bucket.blob(preprocessed_current_data_with_predictions_file)

        # Upload the CSV file
        blob.upload_from_filename(os.path.join(curr_dir, os.pardir, data_folder, preprocessed_current_data_file), content_type="text/csv")

        logging.info("Preprocessed current data with predictions was successfully uploaded to Google bucket.")

    except:
        logging.error("Failed to upload preprocessed current data with predictions to Google bucket!")
        sys.exit(1)


# The preprocess.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['upload_predictions']

if __name__ == "__main__":
    upload_predictions(**params)    