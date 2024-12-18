import logging
import yaml
import os
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO)

# Current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def preprocess(**kwargs):
    
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

    # One-hot transformation
    categorical_columns =  Data.select_dtypes(exclude=['int', 'float']).columns
    if len(categorical_columns) > 0:
        Data = pd.get_dummies(Data, categorical_columns, drop_first=False)
        logging.info('One-hot transform applied for the following columns: {}.'.format(', '.join(categorical_columns)))
    else:
        logging.info('No categorical columns detected. One-hot transform is not applied.')
        
    # Save the preprocessed file
    try:
        Data.to_csv(os.path.join(curr_dir, os.pardir, kwargs['output_file_path']), index=False)
        logging.info("The preprocessed CSV file was successfully saved.")
    except:
        logging.error("The preprocessed CSV file was not saved!")
        sys.exit(1)

    del(Data)

# The preprocess.py parameters from the params.yaml file
params = yaml.safe_load(open(os.path.join(curr_dir, '..', "params.yaml")))['preprocess']

if __name__ == "__main__":
    preprocess(**params)