import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)
logging.info('Loading Python libraries ...')

import yaml
import os
import sys
import pandas as pd

import mlflow

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import json

from utils.mlflow_utils import configure_mlflow

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def preprocess(**kwargs):

    configure_mlflow()
    
    input_folder = kwargs['input_folder']
    input_file = kwargs['input_file']
    keep_duplicates = kwargs['keep_duplicates']
    rename_map = kwargs['rename_map']
    current_subset_size = kwargs["current_subset_size"] 
    prob_coeff = kwargs['prob_coeff']
    column_impacting_the_split = kwargs["column_impacting_the_split"]
    time_period_column_name = kwargs["time_period_column_name"]
    onehot_name_dictionary_file_path = kwargs['onehot_name_dictionary_file_path']
    output_reference_data_file_path = kwargs['output_reference_data_file_path']
    output_current_data_file_path = kwargs['output_current_data_file_path']


    # Initiate logging
    logging.basicConfig(level=logging.INFO)

    # Download the input file into a dataframe
    Data = pd.read_csv(os.path.join(curr_dir, os.pardir, input_folder, input_file))
    logging.info("The data has {} rows and {} columns".format(*Data.shape))

    # Remove all rows with NaNs (if any)
    row_has_NaN = Data.isnull().any(axis=1)
    total_NaNs = row_has_NaN.sum()
    if total_NaNs:
        Data = Data[~row_has_NaN]
        logging.info("Total of {} rows with missing data were deleted.".format(total_NaNs))
    else:
        logging.info("No missing data points were found.")

    # Drop the duplicates (or not)
    if  keep_duplicates == False:
        Data = Data[~Data.duplicated(list(Data.columns))]
        logging.info("Duplicates removed (if any).")

    # Rename the columns
    try:
        Data.rename(columns=rename_map, inplace=True)
        logging.info("Columns renamed.")
    except:
        logging.error("Name mismatch in the column renaming dictionary!")
        sys.exit(1)

    # Split the file into the 'reference and 'current' parts. This will be used by EvidentlyAI later on
    dataset_size = Data.shape[0]
    # Probabilities used to split the data set to the 'reference and the 'current' subsets
    # For prob_coeff=0 the probabilities to fall into either dataset are the same
    probabilities = np.exp(prob_coeff * ((np.arange(dataset_size, dtype=float) / (dataset_size - 1))))
    # Normalization
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(Data.sort_values(column_impacting_the_split, ascending=True).index, 
                                       size=int(current_subset_size * dataset_size), p=probabilities, replace=False)
    # Add the split column values
    Data[time_period_column_name] = pd.Series(Data.index.isin(sampled_indices)).map({True: 'current', False: 'reference'})
    logging.info("Dataset split into the current and the current subsets, {} and {} in size respectively.".format(dataset_size, sampled_indices.size))

    # Start an MLflow run
    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', "Preprocess: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))

        # Add graphs for all of the features
        for idx, col, t in Data.drop(time_period_column_name, axis=1).dtypes.reset_index().reset_index().values:
            fig, axs = plt.subplots(2, 1, figsize=(12, 6))
            # All colors are from the same cmap
            color = plt.colormaps.get_cmap('Dark2')(idx / Data.columns.size)
            for i, tp in zip([0,1], ['reference','current']):

                x, y = zip(*Data[Data[time_period_column_name]==tp][col].value_counts().sort_index().reset_index().values)

                # Calculate optimal bar width
                total_space = 1.0  # Total space allocated for one bar + one gap
                bar_width = max(0.009, total_space * 0.8 / len(x)) # Use 80% of space for bars

                # Create the bar graph
                axs[i].bar(x, y, color=color, align='center', edgecolor='black', width=bar_width)

                # Adjust x-axis limits to ensure bars are fully visible
                if t!='object':
                    if tp=='reference':
                        x_lim_min = x[0] - bar_width / 2
                        x_lim_max = x[-1] + bar_width / 2
                    axs[i].set_xlim(x_lim_min, x_lim_max)

                # Rotate the labels, if there are too many of them
                if t=='object' and len(set(x)) > 5:
                    axs[i].tick_params(axis='x', labelrotation=60)

                # Add custom ticks for 0 and 1
                if set(x).issubset(set([0., 1.])):
                    axs[i].set_xticks([0, 1], ['No', 'Yes'])

                axs[i].set_title("{} ({})".format(col, tp), fontsize=16)
                axs[i].set_ylabel('Count')

            plt.tight_layout()
            mlflow.log_figure(fig, "{}.png".format(col))
            plt.close(fig)
        del(idx, col, t, color, i, tp, total_space, bar_width)

    logging.info("{} data histograms (one per column) were added to the MLFlow.".format(len(Data.columns)))
    

    # One-hot transformation
    # All string columns, except the current/current column
    categorical_columns =  Data.drop(time_period_column_name, axis=1).select_dtypes(exclude=['int', 'float']).columns
    # All numerical columns
    non_categorical_columns =  Data.select_dtypes(include=['int', 'float']).columns
    # Dictionary to keep track of the name changes after pd.get_dummies
    onehot_name_dictionary = dict()
    # Add the non-categorical columns to the dictionary
    for col in non_categorical_columns:
        onehot_name_dictionary[col] = col
    # Convert categorical data with the one-hot transform and add the categorical columns to the dictionary
    if len(categorical_columns) > 0:
        Data = pd.get_dummies(Data, columns=categorical_columns, drop_first=False, prefix=None)
        for col_old in categorical_columns:
            for col_new in Data.columns:
                if col_new.startswith(col_old):
                    onehot_name_dictionary[col_new] = col_old
        logging.info('One-hot encoding was applied for the following columns: {}.'.format(', '.join(categorical_columns)))
    else:
        logging.info('No categorical columns detected. One-hot encoding is not applied.')

    # Save the names dictionary which relates the columnns before and after pd.get_dummies
    try:
        with open(os.path.join(curr_dir, os.pardir, onehot_name_dictionary_file_path), 'w') as json_file:
            json.dump(onehot_name_dictionary, json_file)
        logging.info("The column names dictionary was successfully saved.")
    except:
        logging.error("The column names dictionary was not saved!")
        sys.exit(1)

    # Save the 'reference' preprocessed file
    try:
        Data.query("{}=='reference'".format(time_period_column_name)).drop(time_period_column_name, axis=1).to_csv(os.path.join(curr_dir, 
                                                                                            os.pardir, output_reference_data_file_path), index=False)
        logging.info("The preprocessed reference data CSV file was successfully saved.")
    except:
        logging.error("The preprocessed reference data CSV file was not saved!")
        sys.exit(1)

    # Save the 'current' preprocessed file
    try:
        Data.query("{}=='current'".format(time_period_column_name)).drop(time_period_column_name, axis=1).to_csv(os.path.join(curr_dir, 
                                                                                            os.pardir, output_current_data_file_path), index=False)
        logging.info("The preprocessed current data CSV file was successfully saved.")
    except:
        logging.error("The preprocessed current data CSV file was not saved!")
        sys.exit(1)

    del(Data)

# The preprocess.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['preprocess']

if __name__ == "__main__":
    preprocess(**params)