import os
from pyspark.sql import SparkSession

HIVE_DB = "health_insurance"
FEATURE_TABLE = "claim_features"
ANALYTICS_OUTPUT = "/opt/spark-apps/outputs/analytics"


def build_session():
    return (
        SparkSession.builder
        .appName("HealthInsuranceSQLAnalytics")
        .config("spark.sql.warehouse.dir", "hdfs://hdfs-namenode:9000/user/hive/warehouse")
        .config("spark.sql.shuffle.partitions", "8")
        .enableHiveSupport()
        .getOrCreate()
    )


QUERIES = {
    "region_fraud_patterns": f"""
        SELECT
            region,
            COUNT(*)                                                          AS total_claims,
            SUM(CAST(is_fraud AS INT))                                        AS fraud_claims,
            ROUND(SUM(CAST(is_fraud AS INT)) * 100.0 / COUNT(*), 2)          AS fraud_rate_pct,
            ROUND(AVG(claim_amount), 2)                                       AS avg_claim_amount,
            ROUND(MAX(claim_amount), 2)                                       AS max_claim_amount,
            ROUND(AVG(anomaly_score_proxy), 3)                                AS avg_risk_score
        FROM {HIVE_DB}.{FEATURE_TABLE}
        GROUP BY region
        ORDER BY fraud_rate_pct DESC
    """,

    "top_risk_feature_comparison": f"""
        SELECT
            ROUND(AVG(CASE WHEN is_fraud = true THEN anomaly_score_proxy END), 3)  AS avg_risk_score_fraud,
            ROUND(AVG(CASE WHEN is_fraud = false THEN anomaly_score_proxy END), 3) AS avg_risk_score_normal,
            ROUND(AVG(CASE WHEN is_fraud = true THEN charge_per_age END), 2)       AS avg_charge_per_age_fraud,
            ROUND(AVG(CASE WHEN is_fraud = false THEN charge_per_age END), 2)      AS avg_charge_per_age_normal,
            ROUND(AVG(CASE WHEN is_fraud = true THEN log_claim_amount END), 4)     AS avg_log_claim_fraud,
            ROUND(AVG(CASE WHEN is_fraud = false THEN log_claim_amount END), 4)    AS avg_log_claim_normal,
            SUM(CASE WHEN is_fraud = true AND smoker_obese_flag = 1 THEN 1 ELSE 0 END) AS fraud_smoker_obese,
            SUM(CASE WHEN is_fraud = true AND rapid_resubmission = 1 THEN 1 ELSE 0 END) AS fraud_rapid_resubmit
        FROM {HIVE_DB}.{FEATURE_TABLE}
    """,

    "claim_distribution_bmi_age": f"""
        SELECT
            bmi_category,
            age_group,
            COUNT(*)                              AS claim_count,
            ROUND(AVG(claim_amount), 2)           AS avg_claim,
            ROUND(MIN(claim_amount), 2)           AS min_claim,
            ROUND(MAX(claim_amount), 2)           AS max_claim,
            SUM(CAST(is_fraud AS INT))            AS fraud_count,
            ROUND(SUM(CAST(is_fraud AS INT)) * 100.0 / COUNT(*), 2) AS fraud_pct
        FROM {HIVE_DB}.{FEATURE_TABLE}
        GROUP BY bmi_category, age_group
        ORDER BY avg_claim DESC
    """,

    "suspicious_high_value_claims": f"""
        SELECT
            claim_id,
            patient_id,
            age,
            ROUND(bmi, 1)             AS bmi,
            ROUND(claim_amount, 2)    AS claim_amount,
            procedure_code,
            provider_id,
            region,
            days_since_last_claim,
            ROUND(anomaly_score_proxy, 2) AS risk_score,
            is_fraud
        FROM {HIVE_DB}.{FEATURE_TABLE}
        WHERE claim_amount > 25000
          AND (rapid_resubmission = 1 OR young_high_procedure_flag = 1 OR provider_high_volume_flag = 1)
        ORDER BY claim_amount DESC
        LIMIT 50
    """,

    "provider_anomaly_summary": f"""
        SELECT
            provider_id,
            COUNT(*)                                                           AS total_claims,
            SUM(CAST(is_fraud AS INT))                                         AS confirmed_fraud,
            ROUND(SUM(CAST(is_fraud AS INT)) * 100.0 / COUNT(*), 2)           AS fraud_rate_pct,
            ROUND(AVG(claim_amount), 2)                                        AS avg_claim,
            ROUND(MAX(claim_amount), 2)                                        AS max_claim,
            ROUND(AVG(anomaly_score_proxy), 3)                                 AS avg_risk_score,
            SUM(rapid_resubmission)                                            AS rapid_resubmission_count
        FROM {HIVE_DB}.{FEATURE_TABLE}
        GROUP BY provider_id
        HAVING COUNT(*) > 3
        ORDER BY confirmed_fraud DESC, avg_risk_score DESC
        LIMIT 25
    """,

    "explainability_flagged_claims": f"""
        SELECT
            claim_id,
            age,
            bmi_category,
            age_group,
            procedure_risk,
            region,
            ROUND(claim_amount, 2)          AS claim_amount,
            ROUND(anomaly_score_proxy, 2)   AS risk_score,
            smoker_obese_flag,
            rapid_resubmission,
            young_high_procedure_flag,
            provider_high_volume_flag,
            claim_above_region_p95,
            is_fraud
        FROM {HIVE_DB}.{FEATURE_TABLE}
        WHERE anomaly_score_proxy >= 4
        ORDER BY anomaly_score_proxy DESC, claim_amount DESC
        LIMIT 100
    """,

    "procedure_code_fraud_analysis": f"""
        SELECT
            procedure_code,
            procedure_description,
            procedure_risk,
            COUNT(*)                                                          AS total_claims,
            SUM(CAST(is_fraud AS INT))                                        AS fraud_claims,
            ROUND(SUM(CAST(is_fraud AS INT)) * 100.0 / COUNT(*), 2)          AS fraud_rate_pct,
            ROUND(AVG(claim_amount), 2)                                       AS avg_claim_amount
        FROM {HIVE_DB}.{FEATURE_TABLE}
        GROUP BY procedure_code, procedure_description, procedure_risk
        ORDER BY fraud_rate_pct DESC
    """,
}


def run_analytics(spark):
    os.makedirs(ANALYTICS_OUTPUT, exist_ok=True)
    results = {}

    for query_name, sql in QUERIES.items():
        print(f"\nExecuting: {query_name}")
        try:
            result = spark.sql(sql)
            result.show(20, truncate=False)
            out_path = f"{ANALYTICS_OUTPUT}/{query_name}"
            result.coalesce(1).write.mode("overwrite").option("header", "true").csv(out_path)
            results[query_name] = result.count()
            print(f"  Rows returned: {results[query_name]} | Saved: {out_path}")
        except Exception as exc:
            print(f"  Query failed: {exc}")

    return results


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Running analytics on {HIVE_DB}.{FEATURE_TABLE}")
    results = run_analytics(spark)

    print(f"\nAnalytics complete. Outputs in {ANALYTICS_OUTPUT}")
    for name, count in results.items():
        print(f"  {name}: {count} rows")

    spark.stop()


if __name__ == "__main__":
    main()
