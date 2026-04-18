from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, log1p, count, avg, stddev,
    max as spark_max, min as spark_min,
    round as spark_round, lit, expr, broadcast,
    percentile_approx
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

HDFS_RAW = "hdfs://hdfs-namenode:9000/user/health/raw_claims"
HDFS_FEATURES = "hdfs://hdfs-namenode:9000/user/health/features"
HIVE_DB = "health_insurance"
FEATURE_TABLE = "claim_features"
PROVIDER_STATS_TABLE = "provider_stats"


def build_session():
    return (
        SparkSession.builder
        .appName("HealthInsuranceBatchETL")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.warehouse.dir", "hdfs://hdfs-namenode:9000/user/hive/warehouse")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )


def load_raw(spark):
    df = spark.read.parquet(HDFS_RAW)
    print(f"Loaded {df.count()} raw records from {HDFS_RAW}")
    return df


def build_provider_stats(df):
    return (
        df.groupBy("provider_id")
        .agg(
            count("*").alias("provider_total_claims"),
            avg("claim_amount").alias("provider_avg_claim"),
            spark_max("claim_amount").alias("provider_max_claim"),
            avg(col("is_fraud").cast("double")).alias("provider_fraud_rate"),
        )
    )


def compute_region_claim_stats(df):
    return (
        df.groupBy("region")
        .agg(
            avg("claim_amount").alias("region_avg_claim"),
            stddev("claim_amount").alias("region_stddev_claim"),
            percentile_approx("claim_amount", 0.95).alias("region_p95_claim"),
        )
    )


def engineer_features(df, provider_stats, region_stats):
    df = df.join(broadcast(provider_stats), on="provider_id", how="left")
    df = df.join(broadcast(region_stats), on="region", how="left")

    df = df.withColumn(
        "charge_per_age",
        spark_round(col("claim_amount") / (col("age").cast(DoubleType()) + 1.0), 4),
    )

    df = df.withColumn(
        "bmi_category",
        when(col("bmi") < 18.5, "underweight")
        .when(col("bmi") < 25.0, "normal")
        .when(col("bmi") < 30.0, "overweight")
        .otherwise("obese"),
    )

    df = df.withColumn(
        "age_group",
        when(col("age") < 30, "young")
        .when(col("age") < 50, "middle")
        .when(col("age") < 65, "senior")
        .otherwise("elderly"),
    )

    df = df.withColumn(
        "procedure_risk",
        when(col("procedure_code").isin("27447", "43239", "70553"), "high")
        .when(col("procedure_code").isin("93000", "71046", "80053"), "medium")
        .otherwise("low"),
    )

    df = df.withColumn(
        "smoker_obese_flag",
        when((col("smoker") == True) & (col("bmi") >= 30.0), 1).otherwise(0).cast(IntegerType()),
    )

    df = df.withColumn(
        "high_value_claim",
        when(col("claim_amount") > 20000, 1).otherwise(0).cast(IntegerType()),
    )

    df = df.withColumn(
        "rapid_resubmission",
        when(col("days_since_last_claim") <= 7, 1).otherwise(0).cast(IntegerType()),
    )

    df = df.withColumn(
        "young_high_procedure_flag",
        when((col("age") < 30) & (col("procedure_code") == "27447"), 1).otherwise(0).cast(IntegerType()),
    )

    df = df.withColumn(
        "provider_high_volume_flag",
        when(col("provider_id") == "PRV9999", 1).otherwise(0).cast(IntegerType()),
    )

    df = df.withColumn("log_claim_amount", log1p(col("claim_amount")))

    df = df.withColumn(
        "claim_zscore_region",
        when(col("region_stddev_claim") > 0,
             (col("claim_amount") - col("region_avg_claim")) / col("region_stddev_claim"))
        .otherwise(lit(0.0)),
    )

    df = df.withColumn(
        "claim_above_region_p95",
        when(col("claim_amount") > col("region_p95_claim"), 1).otherwise(0).cast(IntegerType()),
    )

    df = df.withColumn(
        "provider_claim_deviation",
        when(col("provider_avg_claim").isNotNull(),
             spark_round(col("claim_amount") / (col("provider_avg_claim") + 1.0), 4))
        .otherwise(lit(1.0)),
    )

    df = df.withColumn(
        "anomaly_score_proxy",
        (
            col("high_value_claim") * 2
            + col("smoker_obese_flag")
            + col("rapid_resubmission") * 3
            + col("young_high_procedure_flag") * 2
            + col("provider_high_volume_flag") * 2
            + col("claim_above_region_p95")
        ).cast(DoubleType()),
    )

    return df


def encode_categoricals(df):
    indexers = [
        StringIndexer(inputCol="sex", outputCol="sex_idx", handleInvalid="keep"),
        StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep"),
        StringIndexer(inputCol="bmi_category", outputCol="bmi_cat_idx", handleInvalid="keep"),
        StringIndexer(inputCol="age_group", outputCol="age_group_idx", handleInvalid="keep"),
        StringIndexer(inputCol="procedure_risk", outputCol="procedure_risk_idx", handleInvalid="keep"),
    ]
    pipeline = Pipeline(stages=indexers)
    return pipeline.fit(df).transform(df)


def persist_features(df, spark):
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DB}")

    df.write.mode("overwrite").format("parquet").saveAsTable(f"{HIVE_DB}.{FEATURE_TABLE}")
    print(f"Feature table saved: {HIVE_DB}.{FEATURE_TABLE}")

    df.write.mode("overwrite").parquet(HDFS_FEATURES)
    print(f"Parquet written: {HDFS_FEATURES}")


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    raw = load_raw(spark)

    provider_stats = build_provider_stats(raw)
    region_stats = compute_region_claim_stats(raw)

    featured = engineer_features(raw, provider_stats, region_stats)
    encoded = encode_categoricals(featured)

    print(f"Features engineered. Total columns: {len(encoded.columns)}")
    encoded.printSchema()

    persist_features(encoded, spark)

    fraud_count = encoded.filter(col("is_fraud") == True).count()
    total = encoded.count()
    print(f"Dataset: {total} records, {fraud_count} fraud ({round(fraud_count*100/total,2)}%)")

    spark.stop()
    print("Batch ETL complete.")


if __name__ == "__main__":
    main()
