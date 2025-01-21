import requests
import pandas as pd
import json

# Define the URL
url = 'https://real-time-payments-api.herokuapp.com/current-transactions/'

# Make a GET request to retrieve the data
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the response JSON data
    data = response.json()
    print(json.loads(data))  # Print the data or do something with it)
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")