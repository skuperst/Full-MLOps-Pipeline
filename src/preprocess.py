import logging
import yaml
import os
import sys
import pandas as pd

import mlflow

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import json

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

from utils.mlflow_utils import configure_mlflow

experiment_name = configure_mlflow()

def preprocess(**kwargs):
    
    # Initiate logging
    logging.basicConfig(level=logging.INFO)

    # Download the input file into a dataframe
    Data = pd.read_csv(os.path.join(curr_dir, os.pardir, kwargs['input_folder'], kwargs['input_file']))
    logging.info("The data has {} rows and {} columns".format(*Data.shape))

    # Rename the columns
    try:
        Data.rename(columns=kwargs['rename_map'], inplace=True)
        logging.info("Columns renamed.")
    except:
        logging.error("Name mismatch in the column renaming dictionary!")
        sys.exit(1)

    # Remove all rows with NaNs (if any)
    row_has_NaN = Data.isnull().any(axis=1)
    total_NaNs = row_has_NaN.sum()
    if total_NaNs:
        Data = Data[~row_has_NaN]
        logging.info("Total of {} rows with missing data were deleted.".format(total_NaNs))
    else:
        logging.info("No missing data points were found.")


    # Drop the duplicates (or not)
    if kwargs['keep_duplicates'] == False:
        Data = Data[~Data.duplicated(list(Data.columns))]
        logging.info("Duplicates removed (if any).")


    # Start an MLflow run
    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', "Preprocess: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))

        for idx, col, t in Data.dtypes.reset_index().reset_index().values:
            plt.figure(figsize=(7, 4))
            color = plt.colormaps.get_cmap('Dark2')(idx / Data.columns.size)
            if t == 'object':
                plt.bar(*zip(*Data[col].value_counts().reset_index().values), color=color)
                if Data[col].nunique() > 5:
                    plt.xticks(rotation = 60)
            else:
                plt.hist(Data[col], bins = 100, color=color)
            plt.title(col, fontsize=16)
            plt.ylabel('Count')
            if set(Data[col]).issubset(set([0., 1.])):
                # Remove all ticks and labels
                plt.xticks([])  # Remove x-axis ticks
                plt.yticks([])  # Remove y-axis ticks
                plt.tick_params(bottom=False, left=False)  # Remove tick marks
                # Add custom ticks for 0 and 1
                plt.xticks([0, 1], ['No', 'Yes'])

            mlflow.log_figure(plt.gcf(), "{}.png".format(col))
        plt.close()
        del(idx, col, t)

    logging.info("{} data histograms (one per column) were added to the MLFlow.".format(len(Data.columns)))

    # One-hot transformation
    categorical_columns =  Data.select_dtypes(exclude=['int', 'float']).columns
    # Dictionary to keep track of the name changes after pd.get_dummies
    onehot_name_dictionary = dict()
    if len(categorical_columns) > 0:
        Data = pd.get_dummies(Data, categorical_columns, drop_first=False)
        for col_old in categorical_columns:
            for col_new in Data.columns:
                if col_new.startswith(col_old):
                    onehot_name_dictionary[col_new] = col_old
        logging.info('One-hot transform was applied for the following columns: {}.'.format(', '.join(categorical_columns)))
    else:
        logging.info('No categorical columns detected. One-hot transform is not applied.')

    # Save the names dictionary which relates the columnns before and after pd.get_dummies
    try:
        with open(os.path.join(curr_dir, os.pardir, kwargs['onehot_name_dictionary_file_path']), 'w') as json_file:
            json.dump(onehot_name_dictionary, json_file)
        logging.info("The column names dictionary was successfully saved.")
    except:
        logging.error("The column names dictionary was not saved!")
        sys.exit(1)

    # Save the preprocessed file
    try:
        Data.to_csv(os.path.join(curr_dir, os.pardir, kwargs['output_file_path']), index=False)
        logging.info("The preprocessed CSV file was successfully saved.")
    except:
        logging.error("The preprocessed CSV file was not saved!")
        sys.exit(1)

    del(Data)

# The preprocess.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['preprocess']

if __name__ == "__main__":
    preprocess(**params)