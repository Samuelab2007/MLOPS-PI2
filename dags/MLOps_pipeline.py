from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops_user',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Pipeline MLOps con Airflow',
    schedule_interval='@hourly',  # se ejecuta cada hora
    catchup=False,
    tags=['mlops', 'airflow']
) as dag:

    load_data = BashOperator(
        task_id='load_data',
        bash_command='python3 /home/usuario/scripts/load_data.py',
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /home/usuario/scripts/train_model.py',
    )

    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='python3 /home/usuario/scripts/evaluate_model.py',
    )

    # Dependencias: load_data -> train_model -> evaluate_model
    load_data >> train_model >> evaluate_model