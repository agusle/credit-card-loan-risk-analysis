"""dataset_download

This script allows the user to download the dataset components from AWS S3 buckets. 

It requires a .env file with valid access keys so it can take AWS credentials 
to setup the enviroment and download dataset to "data" folder. In addition to doing that,
prints the existing buckets on the filtered destination.

Usage
-----
    python dataset_download.py

Returns
-------
    Download dataset on "Data" folder.
"""

# Python standard Library
import os

# Third party libraries
import boto3
from cloudpathlib import CloudPath
from dotenv import load_dotenv


# look for a .env file and if it finds one, it will load the environment variables
load_dotenv()

# fetch credentials from env variables
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

# setup a AWS S3 client/resource
s3 = boto3.resource(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# point the resource at the existing bucket
bucket = s3.Bucket("anyoneai-datasets")

# print all object names found in the bucket
print("Existing buckets:")
for file in bucket.objects.filter(Prefix="credit-data-2010"):
    print(file)

# download dataset
dataset = CloudPath("s3://anyoneai-datasets/credit-data-2010/")
dataset.download_to("data")
