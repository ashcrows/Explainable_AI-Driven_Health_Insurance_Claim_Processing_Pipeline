#!/usr/bin/env bash
set -e

SPARK_SUBMIT="docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.executor.memory=512m \
  --conf spark.driver.memory=512m"

echo "========================================"
echo " Batch Pipeline — 5-step execution"
echo "========================================"
echo ""

docker exec spark-master bash -c "mkdir -p \
  /opt/spark-apps/outputs/metrics \
  /opt/spark-apps/outputs/predictions \
  /opt/spark-apps/outputs/plots \
  /opt/spark-apps/outputs/benchmarks \
  /opt/spark-apps/outputs/analytics"

echo "[1/5] Batch Feature Engineering..."
$SPARK_SUBMIT /opt/spark-apps/batch_feature_engineering.py
echo "Feature engineering complete."
echo ""

echo "[2/5] ML Model Training (GBT + Logistic Regression with 3-fold CV)..."
echo "Note: This step takes 5-15 min depending on data volume."
$SPARK_SUBMIT /opt/spark-apps/ml_training.py
echo "ML training complete."
echo ""

echo "[3/5] SHAP Explainability..."
$SPARK_SUBMIT /opt/spark-apps/shap_explainability.py
echo "SHAP analysis complete."
echo ""

echo "[4/5] SQL Analytics..."
$SPARK_SUBMIT /opt/spark-apps/sql_analytics.py
echo "Analytics complete."
echo ""

echo "[5/5] Benchmarking..."
$SPARK_SUBMIT /opt/spark-apps/benchmarking.py
echo "Benchmarking complete."
echo ""

echo "========================================"
echo " All batch jobs finished."
echo "========================================"
echo ""
echo "Outputs written to:"
echo "  outputs/metrics/      — training_metrics.json, shap_global_importance.json"
echo "  outputs/predictions/  — predictions_sample.csv, shap_values_sample.csv"
echo "  outputs/plots/        — SHAP beeswarm, bar, dependence plots"
echo "  outputs/analytics/    — CSV results per SQL query"
echo "  outputs/benchmarks/   — benchmark_results.csv"
echo ""
