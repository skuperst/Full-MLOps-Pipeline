import logging
# Initiate logging
logging.basicConfig(level=logging.INFO)

logging.info('Loading Python libraries ...')

# import sys
# import pandas as pd
import os
import pickle
import sys
import json
import yaml
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

logging.info('Done!')

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['flask_api']

model_file_path = params['model_file_path']
flask_dict_file_path = params['flask_dict_file_path']
onehot_name_dictionary_file_path = params['onehot_name_dictionary_file_path']
test_X_file_path = params['test_X_file_path']

# Load the trained model
model = pickle.load(open(model_file_path,'rb'))
# Load the dictionary containing the information about the html file outlook
flask_dict = json.load(open(os.path.join(curr_dir, os.pardir, flask_dict_file_path)))

# Load the dictionary saved earlier at the preprocess stage and containing the info regarding the get_dummies transformation
onehot_column_name_dictionary = json.load(open(os.path.join(curr_dir, os.pardir, onehot_name_dictionary_file_path)))
# Convert it to a panda's Series object
onehot_column_name_series = pd.Series(onehot_column_name_dictionary)
# Extract the categorical columns (they appear more than once in the series)
categorical_columns = onehot_column_name_series.value_counts()[onehot_column_name_series.value_counts()>1].index

# The full list of column in the test dataset. 
test_dataset_columns = pd.read_csv(os.path.join(curr_dir, os.pardir, test_X_file_path), nrows=0).columns

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def form_handler():

    selected_values= dict()

    for key in flask_dict.keys():
        selected_values[key] = ''
    output = None

    if request.method == "POST":
        if "reset" in request.form:  # Handle Reset button
            return redirect(url_for("form_handler"))  # Clear inputs and output by redirecting

        # Retrieve the values from the form and process
        for key in flask_dict.keys():
            selected_values[key] = request.form.get(key)
        
        # Validate required fields
        if all([selected_values[key]!='' for key in flask_dict.keys()]):

            # Build a dataframe with one row from the input data. The columns in the categorical columns list should stay string. Float or int otherwise.
            data = pd.DataFrame([dict([(col, val if (col in categorical_columns) 
                                             else float(val) if '.' in val else int(val)) for col, val in selected_values.items()])])
            # Apply the get_dummies transform like in preprocess.py
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=False, prefix=None)
            # Add columns that were not created above by get_dummies but should be in a test dataset 
            data = data.reindex(columns=test_dataset_columns, fill_value=False)

            # Prepare the output
            output = 'The employee will <strong>{}</strong>\nwith the probability of {} %.'.format('stay' if model.predict(data)[0]==0 else 'leave',
                                                                                 round(model.predict_proba(data).max() * 100), 1)

    return render_template('ml_interface.html', flask_dict=flask_dict, selected_values=selected_values, output=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)