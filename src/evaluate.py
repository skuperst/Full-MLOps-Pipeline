import logging
import yaml
import os
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

import mlflow

import matplotlib.pyplot as plt
import seaborn as sns

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

from utils.mlflow_utils import configure_mlflow

configure_mlflow()

def evaluate(**kwargs):

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

    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', 'evaluate')

        plt.figure(figsize=(8, 6))
        image_labels = ['False','True'] 
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=image_labels , yticklabels=image_labels[::-1], cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        mlflow.log_figure(plt.gcf(), "Confusion Matrix.png")
        logging.info("The confusion matrix was logged.") 
        plt.close()

        mlflow.log_metric("Out-of-sampe Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-score", f1)
        logging.info("The out-of-sampe scores were logged.")        


# The evaluate.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['evaluate']
if __name__=="__main__":
    evaluate(**params)