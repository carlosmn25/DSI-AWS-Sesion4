import boto3
import os
import pandas as pd
from io import StringIO
from joblib import dump, load

BUCKET_NAME = os.environ['BUCKET_NAME']
s3_client = boto3.client('s3')


def get_df_from_s3(s3_key, bucket_name=BUCKET_NAME):
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    csv_string = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    return df


def get_data(data_set='train', bucket_name=BUCKET_NAME):
    s3_path = f'data/{data_set}.csv'
    df = get_df_from_s3(s3_path, bucket_name)
    y = df.pop('label')
    return df, y


def save_model(model, name, bucket_name=BUCKET_NAME):
    local_path = f'/tmp/{name}.joblib'
    dump(model, local_path)
    
    s3_key = f'models/{name}.joblib'
    s3_client.upload_file(local_path, bucket_name, s3_key)
    os.remove(local_path)

    
def load_model(name, bucket_name=BUCKET_NAME):
    s3_key = f'models/{name}.joblib'
    local_path = f'/tmp/{name}.joblib'
    s3_client.download_file(bucket_name, s3_key, local_path)
    
    model = load(local_path)
    os.remove(local_path)
    
    return model
