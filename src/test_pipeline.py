import shutil
import subprocess
import os, yaml

def test_dvc_repro():

    # Current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # The folder parameters
    params = yaml.safe_load(open(os.path.join(curr_dir, os.pardir, "params.yaml")))['download_current_data']
    data_folder = params['data_folder']
    raw_current_data_file = params['raw_current_data_file']

    # Copy test file into the (fake) current data file
    shutil.copy(os.path.join(curr_dir, os.pardir, data_folder, 'test_data.csv'), 
                os.path.join(curr_dir, os.pardir, data_folder, raw_current_data_file))

    result = subprocess.run(["dvc", "repro", "--ignore", 'upload_predictions', "--force"], stdout=None, stderr=None, text=True)

    assert result.returncode == 0
