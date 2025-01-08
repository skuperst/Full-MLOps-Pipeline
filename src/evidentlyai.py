import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

logging.info('Loading Python libraries ...')

import os
import sys
import yaml
import pickle
from dotenv import load_dotenv

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset, ClassificationPreset

import pandas as pd

import mlflow

from utils.mlflow_utils import configure_mlflow

from datetime import datetime

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def evidentlyai(**kwargs):

    # for key, value in kwargs.items():
    #     globals()[key] = value

    evidentlyai_url = kwargs["evidentlyai_url"]
    evidentlyai_project_name = kwargs["evidentlyai_project_name"]
    evidentlyai_project_description = kwargs["evidentlyai_project_description"]
    reference_data_file_path = kwargs['reference_data_file_path']
    current_data_file_path = kwargs['current_data_file_path']
    model_file_path = kwargs['model_file_path']
    prediction_column = kwargs['prediction_column']
    evidently_htmls_folder_name = kwargs['evidently_htmls_folder_name']

    configure_mlflow()
    
    try:
        # Load the .env file
        load_dotenv(override=True)
        # Access the variables
        EVIDENTLY_TOKEN = os.getenv("EVIDENTLY_TOKEN")
        load_dotenv(override=True)
        EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")
        ws = CloudWorkspace(token=EVIDENTLY_TOKEN, url=evidentlyai_url)
        # Create the project if doesn't already exists (meaning that EVIDENTLY_PROJECT_ID is empty)
        if (EVIDENTLY_PROJECT_ID is None):
            project = ws.create_project(evidentlyai_project_name)
            project.description = evidentlyai_project_description
            project.save()
        else:        
            project = ws.get_project(EVIDENTLY_PROJECT_ID)
        logging.info("The connection to the EvidentlyAI API was successfully established.")
    except:
        logging.error("The connection to the EvidentlyAI API could not be established!")
        sys.exit(1)
            
    # File path to save the html version of the report
    html_folder_path = os.path.join(curr_dir, os.pardir, evidently_htmls_folder_name)
    # Ensure the EvidentlyAI html folder exists
    if not os.path.exists(html_folder_path):
        os.makedirs(html_folder_path)
        logging.info("Created the html files folder.")

    # Load the train model
    model = pickle.load(open(model_file_path,'rb'))

    # Load CURRENT and REFERENCE datasets
    reference_data = pd.read_csv(reference_data_file_path) 
    current_data = pd.read_csv(current_data_file_path) 
    # Load the best training model
    model = pickle.load(open(model_file_path,'rb'))

    # Group the columns into categorical and non-categorical
    categorical_columns = reference_data.select_dtypes(exclude=['int', 'float']).columns
    numerical_columns = reference_data.select_dtypes(include=['int', 'float']).columns

    # Calculate the predictions
    reference_data['prediction'] = model.predict(reference_data.drop('left_or_not', axis=1))
    current_data['prediction'] = model.predict(current_data.drop('left_or_not', axis=1))

    column_mapping = ColumnMapping()
    column_mapping.target = prediction_column
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = list(numerical_columns)
    column_mapping.categorical_features = list(categorical_columns)

    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', "EvidentlyAI: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))
   
        for metric in [DataQualityPreset(), DataDriftPreset(), ClassificationPreset()]: # List of metrics

            # Extract metric's name
            metric_name = metric.type.split(':')[-1]

            try:
                # Define an EvidentlyAI report
                report = Report(metrics=[metric], timestamp=datetime.now())
                # Run the report
                report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
            except:
                logging.error("Failed to create the {} Evidently report!".format(metric_name))
                sys.exit(1)

            # Send this report to EvidentlyAI
            try:
                ws.add_report(project.id, report, include_data=False)
                logging.info("Successfully sent the {} report to the EvidentlyAI cloud project.".format(metric_name))
            except:
                logging.error("Failed to send the {} report to the EvidentlyAI cloud project!".format(metric_name))
                sys.exit(1)
                
            # File path to save the html version of the report
            html_file_path = os.path.join(curr_dir, os.pardir, evidently_htmls_folder_name, '{}.html'.format(metric_name))
            # Save this report
            report.save_html(html_file_path)
        
            mlflow.log_artifact(html_file_path)
            logging.info("The {} Evidently html report was successfully logged.".format(metric_name))

            # Save the the drift data and log it
            if metric_name=='DataDriftPreset':
                csv_file_path = os.path.join(curr_dir, os.pardir, evidently_htmls_folder_name, 'drifts.csv')
                drift_df = report.as_dataframe()['DataDriftTable'][['drift_score','drift_detected']].reset_index().sort_values('drift_score', 
                                                                                                                               ascending=False)
                drift_df.to_csv(csv_file_path,  index=False)
                try:
                    mlflow.log_artifact(csv_file_path)
                    logging.info("The drift data was successfully logged.")
                except:
                    logging.error("Failed to log the drift data!")
                    sys.exit(1)

                # Show a warning if drift detected
                if drift_df.query('drift_detected').shape[0] > 0:
                    logging.warning('Drift detected in column(s): {}'.format(', '.join(drift_df[drift_df['drift_detected']]['column_name'])))
        
        project.save()


# The evidentlyai.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['evidentlyai']

if __name__ == "__main__":
    evidentlyai(**params)