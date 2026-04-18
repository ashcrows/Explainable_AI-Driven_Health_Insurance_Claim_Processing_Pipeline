#!/usr/bin/env bash
set -e

echo "Submitting Spark Structured Streaming job..."
echo "This job runs continuously — it ingests Kafka events into HDFS every 30s."
echo "Run submit_batch.sh in a separate terminal once enough data has accumulated (5+ min)."
echo ""

docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.executor.memory=1g \
  --conf spark.driver.memory=1g \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
  /opt/spark-apps/streaming_ingestion.py

echo ""
echo "Streaming job submitted. Monitor at: http://localhost:8080"
