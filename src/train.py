import logging
import yaml
import os
import sys
import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

import math

import mlflow
from mlflow.models import infer_signature

from datetime import datetime

import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap 

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

from utils.mlflow_utils import configure_mlflow

configure_mlflow()

def train(**kwargs):

    # Initiate logging
    logging.basicConfig(level=logging.INFO)

    # Download the data
    Data = pd.read_csv(os.path.join(curr_dir, os.pardir, kwargs['file_path']))

    # Split the prediction column from the dataframe 
    output_column = kwargs['prediction_column'] 
    X = Data.drop(output_column, axis = 1) 
    y = Data[output_column]

    # Convert objects to categories
    logging.info("Convert object type columns to categories.")
    print(X.columns)
    X = X.apply(lambda col: col.astype('category') if col.dtype == 'object' else col, axis = 0)
    print(X.columns)
    
    logging.info("Splitting the data.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=kwargs['random_state'])
    del(X,y)

    # Save the test dataset
    try:
        X_test.to_csv(os.path.join(curr_dir, os.pardir, kwargs['test_X_file_path']), index=False)
        y_test.to_csv(os.path.join(curr_dir, os.pardir, kwargs['test_y_file_path']), index=False)
        logging.info("The test dataset CSV files were successfully saved.")
    except:
        logging.error("The test dataset CSV files were not saved!")
        sys.exit(1)    
    del(X_test, y_test)

    # Define the classifier
    abc = AdaBoostClassifier(DecisionTreeClassifier())
    # Import the hyperparameters grid
    grid_parameters = {'estimator__max_depth': list(range(*kwargs['max_depth'])),
                       'estimator__min_samples_split': list(range(*kwargs['min_samples_split'])),
                       'n_estimators': list(range(*kwargs['n_estimators'])),
                       'learning_rate': list(np.arange(*kwargs['learning_rate']))}

    total_sets_of_parameters = math.prod([len(v) for v in grid_parameters.values()])
    cv = kwargs['cross_validations']
    logging.info("Runnning the grid search with the total of {} sets of hyperparameters (may take up to {} seconds).".format(
                                                                    total_sets_of_parameters, int(total_sets_of_parameters * 0.1 * cv)))
    # Define the scores (accuracy is still the major one, see below)
    scoring = {'Accuracy': 'accuracy', 'F1-score': 'f1', 'Recall':'recall', 'Precision':'precision'}
    grid_search = GridSearchCV(abc, grid_parameters, scoring=scoring, refit='Accuracy', n_jobs=-1, cv=cv, verbose=0)
    start_time = datetime.now()
    grid_search.fit(X_train,y_train)
    end_time = datetime.now()
    logging.info("The grid search was successfully completed in {} seconds.".format((end_time-start_time).seconds))

    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Create the directory to save the model
    os.makedirs(os.path.join(curr_dir, os.pardir, os.path.dirname(kwargs['model_file_path'])), exist_ok=True)

    # Save the model
    pickle.dump(best_model,open(kwargs['model_file_path'],'wb'))
    logging.info("The highest accuracy model was saved locally.")    

    # Create an input example
    input_example = X_train.head(2) + 0. # Converts int to float

    # Optionally, infer the model signature
    signature = infer_signature(X_train.head(2) + 0., y_train.head(2) + 0.)

    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', "Evaluate: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))
        
        sorted_indices = np.argsort(grid_search.cv_results_['mean_test_Accuracy'])
        for score in scoring.keys():
            title = "In-Sample {} - {}".format(score, "the highest acrosss the GS" if score=="Accuracy" else "when Accuracy is the highest")
            the_value_to_log = round(grid_search.cv_results_['mean_test_{}'.format(score)][sorted_indices[-1]], 5)
            mlflow.log_param(title, the_value_to_log)
            logging.info("The {} value was logged.".format(score))

        # Log the best parameters from GridSearchCV
        mlflow.log_params(grid_search.best_params_)

        for score in sorted(scoring.keys()):
            title = "{}In-Sample {} Across GS Iterations".format("Sorted " if score=="Accuracy" else "",score)
            for i, x in enumerate(grid_search.cv_results_['mean_test_{}'.format(score)][sorted_indices]):
                mlflow.log_metric(title, x, step=i+1)
            logging.info("The {} full grid search results were logged.".format(score))

        logging.info("Logging the model to MLFlow ...")
        mlflow.sklearn.log_model(best_model, 'The Highest Accuracy AdaBoost Model', signature=signature, input_example=input_example)
        logging.info("The highest-accuracy AdaBoost model was logged.")

            
    del(best_model, X_train, y_train)

# The train.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['train']

if __name__ == "__main__":
    train(**params)