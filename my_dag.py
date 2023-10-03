from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import news_ingestion
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_dag',
    default_args=default_args,
    description='A simple DAG to run a Python script on a daily basis',
    schedule_interval=timedelta(days=1),
)

def my_task():
    news_ingestion.main()

task = PythonOperator(
    task_id='my_task',
    python_callable=my_task,
    dag=dag,
)
