import os
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
bucket_name = os.getenv('BUCKET_NAME')
credentials_path = os.path.join(os.environ['HOME'], 'google-credentials.json')

# Set up the Google Cloud Storage client
client = storage.Client.from_service_account_json(credentials_path)

# Connect to the bucket and list the blobs
bucket = client.get_bucket(bucket_name)
blobs = list(bucket.list_blobs())
print(f'Connected to {bucket_name}. Blobs in the bucket:')

# Print all blob names
for blob in blobs:
    print(blob.name)