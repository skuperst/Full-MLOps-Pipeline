import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

logging.info('Loading Python libraries ...')

import yaml
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from huggingface_hub import login, HfApi

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['hugging_face_push']

# Hugging Face repository
hf_repo_id = params['hf_repo_id']
# The model trained at train.py
new_model_file_path = params['new_model_file_path']

if os.path.join(curr_dir, os.pardir, new_model_file_path):
    try:
        # Access .env file
        load_dotenv(override=True)
        login(os.getenv("HF_GITHUB_ACTIONS_TOKEN"))
        api = HfApi()

        api.upload_file(
            path_or_fileobj=new_model_file_path,
            path_in_repo=new_model_file_path.split('/')[-1],
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message='The latest model ()'.format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))

        logging.info("The model was successfully deployed to Hugging Face.")
    except:
            logging.error("Couldn't deploy the model to Hugging Face!")
            sys.exit(1)
else:
    logging.info("There is no new model to upload to Hugging Face.") 