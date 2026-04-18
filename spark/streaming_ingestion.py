from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, to_timestamp, current_timestamp,
    when, trim, lower
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType
)

# NOTE: Inside the Docker network, HDFS NameNode is reachable on port 9000.
# The host-side remap (19000) only applies when connecting from outside Docker.
KAFKA_BOOTSTRAP = "kafka:29092"
KAFKA_TOPIC = "health_insurance_claims"
HDFS_RAW_OUTPUT = "hdfs://hdfs-namenode:9000/user/health/raw_claims"
CHECKPOINT_PATH = "hdfs://hdfs-namenode:9000/user/health/checkpoints/streaming_v1"
TRIGGER_INTERVAL = "30 seconds"

CLAIM_SCHEMA = StructType([
    StructField("claim_id", StringType(), True),
    StructField("patient_id", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("sex", StringType(), True),
    StructField("bmi", DoubleType(), True),
    StructField("children", IntegerType(), True),
    StructField("smoker", BooleanType(), True),
    StructField("region", StringType(), True),
    StructField("claim_amount", DoubleType(), True),
    StructField("procedure_code", StringType(), True),
    StructField("procedure_description", StringType(), True),
    StructField("provider_id", StringType(), True),
    StructField("submission_timestamp", StringType(), True),
    StructField("days_since_last_claim", IntegerType(), True),
    StructField("diagnosis_code", StringType(), True),
    StructField("is_fraud", BooleanType(), True),
])

VALID_REGIONS = ["northeast", "northwest", "southeast", "southwest"]
VALID_SEXES = ["male", "female"]


def build_spark_session():
    return (
        SparkSession.builder
        .appName("HealthInsuranceStreamIngestion")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1"
        )
        .getOrCreate()
    )


def read_kafka_stream(spark):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 5000)
        .load()
    )


def parse_and_validate(raw_stream):
    parsed = (
        raw_stream
        .select(
            from_json(col("value").cast("string"), CLAIM_SCHEMA).alias("d"),
            col("timestamp").alias("kafka_ts"),
            col("offset").alias("kafka_offset"),
            col("partition").alias("kafka_partition"),
        )
        .select("d.*", "kafka_ts", "kafka_offset", "kafka_partition")
    )

    validated = (
        parsed
        .filter(
            col("claim_id").isNotNull()
            & col("patient_id").isNotNull()
            & col("age").between(1, 115)
            & col("bmi").between(10.0, 70.0)
            & (col("claim_amount") > 0)
            & col("region").isNotNull()
        )
        .withColumn("sex", lower(trim(col("sex"))))
        .withColumn(
            "sex",
            when(col("sex").isin(VALID_SEXES), col("sex")).otherwise("unknown"),
        )
        .withColumn("region", lower(trim(col("region"))))
        .withColumn(
            "region",
            when(col("region").isin(VALID_REGIONS), col("region")).otherwise("unknown"),
        )
        .withColumn("submission_ts", to_timestamp(col("submission_timestamp")))
        .withColumn("ingestion_ts", current_timestamp())
        .withColumn(
            "claim_amount",
            when(col("claim_amount") > 500000, 500000.0).otherwise(col("claim_amount")),
        )
        .withColumn(
            "bmi",
            when(col("bmi") < 10.0, 10.0)
            .when(col("bmi") > 70.0, 70.0)
            .otherwise(col("bmi")),
        )
        .dropDuplicates(["claim_id"])
        .drop("submission_timestamp")
    )

    return validated


def write_to_hdfs(validated_stream):
    return (
        validated_stream
        .writeStream
        .format("parquet")
        .option("path", HDFS_RAW_OUTPUT)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .partitionBy("region")
        .outputMode("append")
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )


def main():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP}, topic: {KAFKA_TOPIC}")
    raw = read_kafka_stream(spark)
    validated = parse_and_validate(raw)
    query = write_to_hdfs(validated)

    print(f"Streaming query started. Writing to {HDFS_RAW_OUTPUT}")
    print(f"Trigger interval: {TRIGGER_INTERVAL}")
    print("Press Ctrl+C (or stop the container) to terminate.")

    query.awaitTermination()


if __name__ == "__main__":
    main()
