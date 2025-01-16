import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

logging.info('Loading Python libraries ...')

import yaml
import os
import json
import sys
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['hugging_face_push']

# Hugging Face repository
hf_repo_id = params['hf_repo_id']
# The model trained at train.py
model_file_path = params['model_file_path']
# The out-of-sample accuracies
current_accuracy_dict_file = params['current_accuracy_dict_file']
f1_threshold = params['f1_threshold']

current_accuracy_dict_path = open(os.path.join(curr_dir, os.pardir, current_accuracy_dict_file))
current_accuracy_dict = json.load(current_accuracy_dict_path)

if current_accuracy_dict['F1-score'] > f1_threshold:
    try:
        # Access .env file
        load_dotenv(override=True)
        login(os.getenv("HF_GITHUB_ACTIONS_TOKEN"))
        api = HfApi()

        api.upload_file(
            path_or_fileobj=model_file_path,
            path_in_repo=model_file_path.split('/')[-1],
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=', '.join(['{}={} %'.format(name,round(100*score, 2)) for name, score in current_accuracy_dict.items()])
        )
        logging.info("The model was successfully deployed to Hugging Face.")
    except:
        logging.error("Couldn't deploy the model to Hugging Face!")
        sys.exit(1)
else:
    logging.info("The F1-score ({} %) is too low for the Hugging Face deployment.".format(round(100*current_accuracy_dict['F1-score'], 2)))
