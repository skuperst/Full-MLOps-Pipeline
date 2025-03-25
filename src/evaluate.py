import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

from utils.update_check_utils import update_check

if update_check():
    logging.info('Initiating model evaluation.')
else:
    logging.info('No current data downloaded. Skipping evaluation.')     
    exit()

# If there is an update load the heavy Python libraries 
logging.info('Loading Python libraries ...')

import os, yaml
import pickle
import pandas as pd
import json

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import mlflow

import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

import mlflow

from datetime import datetime

from utils.mlflow_utils import configure_mlflow

import math

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate(**kwargs):

    configure_mlflow()
    
    data_folder = kwargs['data_folder']
    test_X_file = kwargs['test_X_file']
    test_y_file = kwargs['test_y_file']
    model_file_path = kwargs['new_model_file_path']
    n_repeats = kwargs['n_repeats']
    random_state = kwargs['random_state']
    onehot_name_dictionary_file = kwargs['onehot_name_dictionary_file']

    evidently_htmls_folder_name = kwargs['evidently_htmls_folder_name']
    drift_file = kwargs['drift_file']


    drift_df = pd.read_csv(os.path.join(curr_dir, os.pardir, evidently_htmls_folder_name, drift_file))
    if drift_df.query('drift_detected').shape[0] > 0:
        logging.info("Data drift detected. The retrained model will be re-evaluated.")
    else:
        logging.info("No data drift detected. The model will be NOT re-evaluated.")
        return
    
    # Load the preprocessed test data
    X_test = pd.read_csv(os.path.join(curr_dir, os.pardir, data_folder, test_X_file))
    y_test = pd.read_csv(os.path.join(curr_dir, os.pardir, data_folder, test_y_file))

    # Load the train model
    model = pickle.load(open(model_file_path,'rb'))

    # Calculate the test set predictions
    y_pred = model.predict(X_test)

    labels = [0, 1]
    # Out-of-sample accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Precision
    precision = precision_score(y_test, y_pred)
    # Recall
    recall = recall_score(y_test, y_pred)
    # F1 score
    f1 = f1_score(y_test, y_pred)

    # Extract feature importances
    permutation_importance_output = permutation_importance(model, X_test, y_test, 
                                                            n_repeats=n_repeats, random_state=random_state)
    # Insert the feature importance info into a dataframe
    importances_df = pd.DataFrame({'Feature': X_test.columns, 'Importnace': permutation_importance_output.importances_mean, 
                                    'STD':permutation_importance_output.importances_std})
    
    # Load the dictionary to restore the categorical column names before the one-hat was applied
    onehot_column_name_dictionary = json.load(open(os.path.join(curr_dir, os.pardir, data_folder, onehot_name_dictionary_file)))
    # Use the dict to restore feature names (i.g. 'department_IT' back to 'department')
    importances_df['Feature'] = importances_df['Feature'].map(onehot_column_name_dictionary)
    # Group the features: sum for importnaces and volatilities
    importances_df = importances_df.groupby('Feature').agg({'Importnace':'sum','STD': lambda s: math.sqrt((s**2).sum())}).reset_index()
    # Sort by importance
    importances_df = importances_df.sort_values('Importnace', ascending=True)

    # Color map for the horizontal bars in the feature importance histogram
    cmap = LinearSegmentedColormap.from_list('Green',["w", "g"], N=256)
    color = cmap((MinMaxScaler().fit_transform(importances_df['Importnace'].values.reshape(-1, 1)).flatten() * 256).astype(int))
    
    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', "Evaluate: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))

        plt.figure(figsize=(12, 4))

        # Plot horizontal bar chart
        plt.barh(y=importances_df['Feature'], width=importances_df['Importnace'], xerr=importances_df['STD'], 
                 capsize=4, align='center', color=color)

        plt.xlabel('Permutation Importance')
        plt.title('Feature Permutation Importance with Standard Deviation')
        plt.tight_layout()

        mlflow.log_figure(plt.gcf(), "Feature Permutation Importance.png")
        plt.close()

        logging.info("The feature permutation importance histogram was logged.") 

        plt.figure(figsize=(6, 6))
        image_labels = ['False','True'] 
        sns.heatmap(cm, annot=True, annot_kws={"fontsize": 18}, fmt='d', xticklabels=image_labels , yticklabels=image_labels[::-1], 
                    cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        mlflow.log_figure(plt.gcf(), "Confusion Matrix.png")
        plt.close()

        logging.info("The confusion matrix was logged.") 

        mlflow.log_metric("Out-of-sample Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-score", f1)

        logging.info("The scores were logged.")


    del(model, X_test, y_test, y_pred)     

# The evaluate.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['evaluate']
if __name__=="__main__":
    evaluate(**params)