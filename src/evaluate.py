import logging
import yaml
import os
import pickle
import pandas as pd
import json

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

import mlflow

import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

import mlflow
from mlflow.models import infer_signature

from datetime import datetime

from utils.mlflow_utils import configure_mlflow

import math

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate(**kwargs):

    configure_mlflow()

    # Initiate logging
    logging.basicConfig(level=logging.INFO)
    
    # Load the preprocessed test data
    X_test = pd.read_csv(os.path.join(curr_dir, os.pardir, kwargs['test_X_file_path']))
    y_test = pd.read_csv(os.path.join(curr_dir, os.pardir, kwargs['test_y_file_path']))

    # Load the train model
    model = pickle.load(open(kwargs['model_file_path'],'rb'))

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
                                                            n_repeats=kwargs['n_repeats'], random_state=kwargs['random_state'])
    # Insert the feature importance info into a dataframe
    importances_df = pd.DataFrame({'Feature': X_test.columns, 'Importnace': permutation_importance_output.importances_mean, 
                                    'STD':permutation_importance_output.importances_std})
    
    # Load the dictionary to restore the categorical column names before the one-hat was applied
    onehot_column_name_dictionary = json.load(open(os.path.join(curr_dir, os.pardir, kwargs['onehot_name_dictionary_file_path'])))
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