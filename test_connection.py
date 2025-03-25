import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

import os
from dotenv import load_dotenv

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset, ClassificationPreset


# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

evidentlyai_url = "https://app.evidently.cloud"
evidentlyai_project_name = "Human Resources"
evidentlyai_project_description = "Use AdaBoost to predict who left the company"

# Load the .env file
load_dotenv(override=True)
# Access the variables
EVIDENTLY_TOKEN = os.getenv("EVIDENTLY_TOKEN")
EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")
ws = CloudWorkspace(token=EVIDENTLY_TOKEN, url=evidentlyai_url)  
# Create the project if doesn't already exists (meaning that EVIDENTLY_PROJECT_ID is empty)
if (EVIDENTLY_PROJECT_ID is None):
    project = ws.create_project(evidentlyai_project_name)
    project.description = evidentlyai_project_description
    project.save()
else:        
    project = ws.get_project(EVIDENTLY_PROJECT_ID)
