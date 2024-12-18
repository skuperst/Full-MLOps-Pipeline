import logging
import yaml
import os
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate(**kwargs):

    # Load the preprocessed test data
    X_test = pd.read_csv(os.path.join(curr_dir, os.pardir, kwargs['test_X_file_path']))
    y_test = pd.read_csv(os.path.join(curr_dir, os.pardir, kwargs['test_y_file_path']))

    # Load the train model
    model = pickle.load(open(kwargs['model_file_path'],'rb'))

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

# The evaluate.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, '..', "params.yaml")))['evaluate']
if __name__=="__main__":
    evaluate(**params)