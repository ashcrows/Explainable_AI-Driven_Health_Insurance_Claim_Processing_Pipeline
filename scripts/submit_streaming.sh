#!/usr/bin/env bash
set -e

echo "Submitting Spark Structured Streaming job..."

docker exec spark-master sh -c '
  mkdir -p /tmp/.ivy/cache /tmp/.ivy/jars &&
  /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.executor.memory=1g \
    --conf spark.driver.memory=1g \
    --conf spark.jars.ivy=/tmp/.ivy \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
    /opt/spark-apps/streaming_ingestion.py
'

echo "Streaming job submitted. Check Spark UI at http://localhost:8080"