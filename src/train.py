import logging, os, yaml
# Initiate logging
logging.basicConfig(level=logging.INFO)

from utils.update_check_utils import update_check

update_check = update_check()
if update_check:

    # If there is an update load the heavy Python libraries 
    logging.info('Loading Python libraries ...')

    import sys
    import pickle

    import pandas as pd
    import numpy as np

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split

    import math
    from datetime import datetime

    import mlflow
    from mlflow.models import infer_signature
    from utils.mlflow_utils import configure_mlflow

    logging.info('Done!')

else:
    from pathlib import Path

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def train(**kwargs):

    output_column = kwargs['prediction_column']
    file = kwargs['file']
    data_folder = kwargs['data_folder']
    random_state=kwargs['random_state']
    test_X_file = kwargs['test_X_file']
    test_y_file = kwargs['test_y_file']
    max_depth = kwargs['max_depth']
    min_samples_split = kwargs['min_samples_split']
    n_estimators = kwargs['n_estimators']
    learning_rate = kwargs['learning_rate']
    model_file_path = kwargs['new_model_file_path']    
    cross_validations = kwargs['cross_validations']

    evidently_htmls_folder_name = kwargs['evidently_htmls_folder_name']
    drift_file = kwargs['drift_file']

    if update_check:
        logging.info('Initiating model training.')
    else:
        logging.info('No current data downloaded. Skipping training with tempty outputs.')
        # Create empty outs file
        Path(os.path.join(curr_dir, os.pardir, data_folder, test_X_file)).touch()
        Path(os.path.join(curr_dir, os.pardir, data_folder, test_y_file)).touch()
        Path(os.path.join(curr_dir, os.pardir, model_file_path)).touch()        
        exit()

    
    drift_df = pd.read_csv(os.path.join(curr_dir, os.pardir, evidently_htmls_folder_name, drift_file))
    if drift_df.query('drift_detected').shape[0] > 0:
        logging.info("Data drift detected. The model will be retrained with the new data.")
    else:
        logging.info("No data drift detected. The model will be  NOT retrained.")
        return

    configure_mlflow()

    # Download the data
    Data = pd.read_csv(os.path.join(curr_dir, os.pardir, data_folder, file))

    # Split the prediction column from the dataframe 
    X = Data.drop(output_column, axis = 1) 
    y = Data[output_column]

    # Convert objects to categories
    X = X.apply(lambda col: col.astype('category') if col.dtype == 'object' else col, axis = 0)
    logging.info("Convert object type columns to categories.")

    logging.info("Splitting the data.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    del(X,y)

    # Save the test datasets
    try:
        X_test.to_csv(os.path.join(curr_dir, os.pardir, data_folder, test_X_file), index=False) 
        y_test.to_csv(os.path.join(curr_dir, os.pardir, data_folder, test_y_file), index=False)
        logging.info("The test dataset CSV files were successfully saved.")
    except:
        logging.error("The test dataset CSV files were not saved!")
        sys.exit(1)    
    del(X_test, y_test)

    # Define the classifier
    abc = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME" )
    # Import the hyperparameters grid
    grid_parameters = {'estimator__max_depth': list(range(*max_depth)),
                       'estimator__min_samples_split': list(range(*min_samples_split)),
                       'n_estimators': list(range(*n_estimators)),
                       'learning_rate': list(np.arange(*learning_rate))}

    total_sets_of_parameters = math.prod([len(v) for v in grid_parameters.values()])
    logging.info("Runnning the grid search with the total of {} sets of hyperparameters (may take up to {} seconds).".format(
                                                                    total_sets_of_parameters, int(total_sets_of_parameters * 0.1 * cross_validations)))
    # Define the scores (accuracy is still the major one, see below)
    scoring = {'Accuracy': 'accuracy', 'F1-score': 'f1', 'Recall':'recall', 'Precision':'precision'}
    grid_search = GridSearchCV(abc, grid_parameters, scoring=scoring, refit='Accuracy', n_jobs=-1, cv=cross_validations, verbose=0)
    start_time = datetime.now()
    grid_search.fit(X_train,y_train)
    end_time = datetime.now()
    logging.info("The grid search was successfully completed in {} seconds.".format((end_time-start_time).seconds))

    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Create the directory to save the model
    os.makedirs(os.path.join(curr_dir, os.pardir, os.path.dirname(model_file_path)), exist_ok=True)

    # Save the model
    pickle.dump(best_model,open(model_file_path,'wb'))
    logging.info("The highest accuracy model was saved locally.")    

    # Create an input example
    input_example = X_train.head(2) + 0. # Converts int to float

    # Optionally, infer the model signature
    signature = infer_signature(X_train.head(2) + 0., y_train.head(2) + 0.)

    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', "Train: {}".format(datetime.now().strftime("%Y/%m/%d (%H:%M)")))
        
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