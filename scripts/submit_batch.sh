#!/usr/bin/env bash
set -euo pipefail

run_spark_job() {
  local script_path="$1"
  local step_name="$2"

  echo "=== ${step_name} ==="

  docker exec spark-master sh -c "
    mkdir -p /tmp/.ivy/cache /tmp/.ivy/jars &&
    /opt/spark/bin/spark-submit \
      --master spark://spark-master:7077 \
      --conf spark.executor.memory=1g \
      --conf spark.driver.memory=1g \
      --conf spark.jars.ivy=/tmp/.ivy \
      ${script_path}
  "

  echo "${step_name} complete."
  echo ""
}

run_spark_job /opt/spark-apps/batch_feature_engineering.py "Step 1: Batch Feature Engineering"
run_spark_job /opt/spark-apps/ml_training.py "Step 2: ML Model Training"
run_spark_job /opt/spark-apps/shap_explainability.py "Step 3: SHAP Explainability"
run_spark_job /opt/spark-apps/sql_analytics.py "Step 4: SQL Analytics"
run_spark_job /opt/spark-apps/benchmarking.py "Step 5: Benchmarking"

echo "All batch jobs finished. Outputs in outputs/ directory."