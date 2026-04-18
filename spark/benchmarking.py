import time
import os
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum as spark_sum

HDFS_FEATURES = "hdfs://hdfs-namenode:9000/user/health/features"
HDFS_RAW = "hdfs://hdfs-namenode:9000/user/health/raw_claims"
BENCHMARK_OUTPUT = "/opt/spark-apps/outputs/benchmarks/benchmark_results.csv"

FIELDS = ["test_name", "num_records", "elapsed_sec", "throughput_rec_per_sec", "latency_ms_per_rec", "notes"]


def build_session(app_name="PipelineBenchmark"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def bench_full_read(spark, path):
    df, elapsed = timed(lambda: spark.read.parquet(path).cache())
    n, _ = timed(lambda: df.count())
    throughput = round(n / elapsed, 1) if elapsed > 0 else 0.0
    latency_ms = round((elapsed / n * 1000) if n > 0 else 0.0, 5)
    return {
        "test_name": "full_read_count",
        "num_records": n,
        "elapsed_sec": round(elapsed, 3),
        "throughput_rec_per_sec": throughput,
        "latency_ms_per_rec": latency_ms,
        "notes": f"Read parquet from {path}",
    }


def bench_filter_select(spark, path):
    df = spark.read.parquet(path)
    def run():
        return df.filter(col("claim_amount") > 10000).select(
            "claim_id", "claim_amount", "region", "is_fraud"
        ).count()
    n, elapsed = timed(run)
    throughput = round(n / elapsed, 1) if elapsed > 0 else 0.0
    return {
        "test_name": "filter_and_projection",
        "num_records": n,
        "elapsed_sec": round(elapsed, 3),
        "throughput_rec_per_sec": throughput,
        "latency_ms_per_rec": round((elapsed / n * 1000) if n > 0 else 0.0, 5),
        "notes": "claim_amount > 10000, select 4 cols",
    }


def bench_aggregation(spark, path):
    df = spark.read.parquet(path)
    def run():
        return df.groupBy("region", "bmi_category").agg(
            count("*").alias("cnt"),
            avg("claim_amount").alias("avg_claim"),
            spark_sum(col("is_fraud").cast("integer")).alias("fraud_count"),
        ).collect()
    result, elapsed = timed(run)
    n = len(result)
    return {
        "test_name": "group_by_aggregation",
        "num_records": n,
        "elapsed_sec": round(elapsed, 3),
        "throughput_rec_per_sec": 0,
        "latency_ms_per_rec": round(elapsed * 1000, 2),
        "notes": "GROUP BY region, bmi_category with 3 aggregations",
    }


def bench_join_operation(spark, path):
    df = spark.read.parquet(path)
    provider_stats = df.groupBy("provider_id").agg(avg("claim_amount").alias("p_avg"))

    def run():
        return df.join(provider_stats, on="provider_id", how="left").count()

    n, elapsed = timed(run)
    throughput = round(n / elapsed, 1) if elapsed > 0 else 0.0
    return {
        "test_name": "join_provider_stats",
        "num_records": n,
        "elapsed_sec": round(elapsed, 3),
        "throughput_rec_per_sec": throughput,
        "latency_ms_per_rec": round((elapsed / n * 1000) if n > 0 else 0.0, 5),
        "notes": "Left join claims with aggregated provider stats",
    }


def bench_scaling_simulation(spark, path):
    df_full = spark.read.parquet(path)
    n_full = df_full.count()
    results = []

    for fraction, label in [(0.25, "1_worker_25pct"), (0.5, "2_workers_50pct"), (1.0, "4_workers_100pct")]:
        sample = df_full.sample(fraction=fraction, seed=42)
        def run(s=sample):
            return s.groupBy("region").agg(
                count("*").alias("cnt"),
                avg("claim_amount").alias("avg_amount"),
            ).collect()
        result, elapsed = timed(run)
        n = int(n_full * fraction)
        throughput = round(n / elapsed, 1) if elapsed > 0 else 0.0
        results.append({
            "test_name": f"scaling_{label}",
            "num_records": n,
            "elapsed_sec": round(elapsed, 3),
            "throughput_rec_per_sec": throughput,
            "latency_ms_per_rec": round((elapsed / n * 1000) if n > 0 else 0.0, 5),
            "notes": f"Scaling simulation at {int(fraction*100)}% data volume",
        })
    return results


def save_results(results):
    os.makedirs(os.path.dirname(BENCHMARK_OUTPUT), exist_ok=True)
    with open(BENCHMARK_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nBenchmark results saved: {BENCHMARK_OUTPUT}")


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    results = []

    print("Running benchmark: full_read_count")
    results.append(bench_full_read(spark, HDFS_FEATURES))

    print("Running benchmark: filter_and_projection")
    results.append(bench_filter_select(spark, HDFS_FEATURES))

    print("Running benchmark: group_by_aggregation")
    results.append(bench_aggregation(spark, HDFS_FEATURES))

    print("Running benchmark: join_provider_stats")
    results.append(bench_join_operation(spark, HDFS_FEATURES))

    print("Running benchmark: scaling simulation (3 variants)")
    results.extend(bench_scaling_simulation(spark, HDFS_FEATURES))

    print("\nBenchmark Summary:")
    print(f"{'Test':<35} {'Records':>10} {'Elapsed(s)':>12} {'Throughput':>14} {'Latency(ms)':>13}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['test_name']:<35} {r['num_records']:>10} "
            f"{r['elapsed_sec']:>12.3f} {r['throughput_rec_per_sec']:>14.1f} "
            f"{r['latency_ms_per_rec']:>13.4f}"
        )

    save_results(results)
    spark.stop()


if __name__ == "__main__":
    main()
