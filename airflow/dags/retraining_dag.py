from datetime import datetime, timedelta
import os
import json
import subprocess

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

SPARK_MASTER = "spark://spark-master:7077"
SPARK_APPS = "/opt/spark-apps"
SPARK_SUBMIT_BASE = (
    "docker exec spark-master "
    "/opt/spark/bin/spark-submit "
    f"--master {SPARK_MASTER} "
    "--conf spark.executor.memory=1g "
    "--conf spark.driver.memory=1g "
)
KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1"
METRICS_PATH = f"{SPARK_APPS}/outputs/metrics/training_metrics.json"
REPORT_PATH = f"{SPARK_APPS}/outputs/airflow_run_report.txt"

DEFAULT_ARGS = {
    "owner": "health_xai_team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 7),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}


def verify_hdfs_data(**context):
    result = subprocess.run(
        [
            "docker", "exec", "hdfs-namenode",
            "hdfs", "dfs", "-count",
            "hdfs://hdfs-namenode:9000/user/health/raw_claims",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"HDFS check failed: {result.stderr.strip()}")

    parts = result.stdout.strip().split()
    num_files = int(parts[1]) if len(parts) >= 2 else 0
    if num_files < 1:
        raise RuntimeError("No parquet files in raw_claims path. Streaming ingestion may not have run.")

    context["ti"].xcom_push(key="hdfs_file_count", value=num_files)
    print(f"HDFS check passed. Files found: {num_files}")


def generate_retraining_report(**context):
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    gbt = metrics.get("gbt", {})
    lr = metrics.get("logistic_regression", {})
    file_count = context["ti"].xcom_pull(task_ids="verify_hdfs_data", key="hdfs_file_count") or "N/A"

    lines = [
        "Automated Retraining Report",
        f"Run timestamp : {datetime.utcnow().isoformat()} UTC",
        f"DAG ID        : health_insurance_weekly_retraining",
        f"Run ID        : {context['run_id']}",
        f"HDFS files    : {file_count}",
        "",
        "GBT Classifier Results",
        f"  AUC-ROC    : {gbt.get('auc_roc', 'N/A')}",
        f"  AUC-PR     : {gbt.get('auc_pr', 'N/A')}",
        f"  F1 Score   : {gbt.get('f1', 'N/A')}",
        f"  Precision  : {gbt.get('precision', 'N/A')}",
        f"  Recall     : {gbt.get('recall', 'N/A')}",
        f"  Accuracy   : {gbt.get('accuracy', 'N/A')}",
        "",
        "Logistic Regression Baseline",
        f"  AUC-ROC    : {lr.get('auc_roc', 'N/A')}",
        f"  F1 Score   : {lr.get('f1', 'N/A')}",
        "",
        "Status: Retraining completed successfully.",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written: {REPORT_PATH}")


with DAG(
    dag_id="health_insurance_weekly_retraining",
    default_args=DEFAULT_ARGS,
    description="Weekly fraud detection model retraining with SHAP explainability",
    schedule_interval="0 2 * * 1",
    catchup=False,
    max_active_runs=1,
    tags=["health-insurance", "fraud-detection", "ml", "xai"],
) as dag:

    verify_data = PythonOperator(
        task_id="verify_hdfs_data",
        python_callable=verify_hdfs_data,
        provide_context=True,
    )

    batch_etl = BashOperator(
        task_id="run_batch_feature_engineering",
        bash_command=(
            f"{SPARK_SUBMIT_BASE} "
            f"{SPARK_APPS}/batch_feature_engineering.py"
        ),
    )

    ml_training = BashOperator(
        task_id="run_ml_training",
        bash_command=(
            f"{SPARK_SUBMIT_BASE} "
            f"{SPARK_APPS}/ml_training.py"
        ),
    )

    shap_explain = BashOperator(
        task_id="run_shap_explainability",
        bash_command=(
            f"{SPARK_SUBMIT_BASE} "
            f"{SPARK_APPS}/shap_explainability.py"
        ),
    )

    sql_analytics = BashOperator(
        task_id="run_sql_analytics",
        bash_command=(
            f"{SPARK_SUBMIT_BASE} "
            f"{SPARK_APPS}/sql_analytics.py"
        ),
    )

    generate_report = PythonOperator(
        task_id="generate_retraining_report",
        python_callable=generate_retraining_report,
        provide_context=True,
    )

    verify_data >> batch_etl >> ml_training >> shap_explain >> sql_analytics >> generate_report
