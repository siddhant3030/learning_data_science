from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.transfers.s3_to_local import S3ToLocalOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalToS3Operator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import zipfile
import pandas as pd

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Data pipeline to unzip, clean, transform, and optimize CSV files',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Define paths and variables
local_dir = '/tmp/data_pipeline/'
os.makedirs(local_dir, exist_ok=True)

bucket_name = 'myawsbuckettest3030'
zip_file_pattern = 'data/2022/04/04/*.zip'
unzip_dir = os.path.join(local_dir, 'unzipped/')
os.makedirs(unzip_dir, exist_ok=True)
cleaned_dir = os.path.join(local_dir, 'cleaned/')
os.makedirs(cleaned_dir, exist_ok=True)

# Task to download zip files from S3
def download_zip_files():
    s3_hook = S3Hook(aws_conn_id='aws_default')
    keys = s3_hook.list_keys(bucket_name=bucket_name, prefix='data/2022/04/04/')
    zip_files = [key for key in keys if key.endswith('.zip')]

    for key in zip_files:
        local_path = os.path.join(local_dir, os.path.basename(key))
        s3_hook.get_key(key, bucket_name).download_file(local_path)

download_zip_task = PythonOperator(
    task_id='download_zip_files',
    python_callable=download_zip_files,
    dag=dag,
)

# Task to unzip files
def unzip_files():
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)

unzip_task = PythonOperator(
    task_id='unzip_files',
    python_callable=unzip_files,
    dag=dag,
)

# Task to clean and transform CSV files
def clean_and_transform_files():
    for root, _, files in os.walk(unzip_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Clean and transform data
                df['Ticker'] = df['Ticker'].str.replace('.NSE', '')
                df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.astype({
                    'Ticker': 'str',
                    'Timestamp': 'datetime64[ns]',
                    'Open': 'float',
                    'High': 'float',
                    'Low': 'float',
                    'Close': 'float',
                    'Volume': 'int',
                })

                # Save cleaned CSV
                cleaned_file_path = os.path.join(cleaned_dir, file)
                df.to_csv(cleaned_file_path, index=False)

clean_and_transform_task = PythonOperator(
    task_id='clean_and_transform_files',
    python_callable=clean_and_transform_files,
    dag=dag,
)

# Task to upload cleaned files to S3
def upload_cleaned_files():
    s3_hook = S3Hook(aws_conn_id='aws_default')
    for root, _, files in os.walk(cleaned_dir):
        for file in files:
            file_path = os.path.join(root, file)
            s3_hook.load_file(
                filename=file_path,
                key=f'cleaned/{file}',
                bucket_name=bucket_name,
                replace=True
            )

upload_task = PythonOperator(
    task_id='upload_cleaned_files',
    python_callable=upload_cleaned_files,
    dag=dag,
)

# Set task dependencies
download_zip_task >> unzip_task >> clean_and_transform_task >> upload_task
