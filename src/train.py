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

import math

logging.basicConfig(level=logging.INFO)

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def train(**kwargs):

    # Download the data
    Data = pd.read_csv(os.path.join(curr_dir, '..', kwargs['file_path']))

    # Split the prediction column from the dataframe 
    output_column = kwargs['prediction_column'] 
    X = Data.drop(output_column, axis = 1) 
    y = Data[output_column]

    # Convert objects to categories
    logging.info("Convert object type columns to categories.")
    X = X.apply(lambda col: col.astype('category') if col.dtype == 'object' else col, axis = 0)

    logging.info("Splitting the data.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=kwargs['random_state'])  

    # Define the classifier
    abc = AdaBoostClassifier(DecisionTreeClassifier())
    # Import the hyperparameters grid
    grid_parameters = {'estimator__max_depth': list(range(*kwargs['max_depth'])),
                       'estimator__min_samples_split': list(range(*kwargs['min_samples_split'])),
                       'n_estimators': list(range(*kwargs['n_estimators'])),
                       'learning_rate': list(np.arange(*kwargs['learning_rate']))}

    total_sets_of_parameters = math.prod([len(v) for v in grid_parameters.values()])
    logging.info("Runnning the grid search with the total of {} sets of hyperparameters (may take up to {} seconds).".format(
                                                                    total_sets_of_parameters, int(total_sets_of_parameters * 0.3)))
    grid_search = GridSearchCV(abc, grid_parameters, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train,y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Create the directory to save the model
    os.makedirs(os.path.join(curr_dir, '..', os.path.dirname(kwargs['model_file_path'])), exist_ok=True)

    # Save the model
    pickle.dump(best_model,open(kwargs['model_file_path'],'wb'))
    logging.info("The best model saved.")    


# The preprocess.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, '..', "params.yaml")))['train']

if __name__ == "__main__":
    train(**params)