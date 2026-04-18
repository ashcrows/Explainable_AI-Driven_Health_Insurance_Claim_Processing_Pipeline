import json
import os
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

HDFS_FEATURES = "hdfs://hdfs-namenode:9000/user/health/features"
GBT_MODEL_PATH = "hdfs://hdfs-namenode:9000/user/health/models/gbt_fraud_v1"
LR_MODEL_PATH = "hdfs://hdfs-namenode:9000/user/health/models/lr_baseline_v1"

METRICS_DIR = "/opt/spark-apps/outputs/metrics"
PREDICTIONS_DIR = "/opt/spark-apps/outputs/predictions"

LABEL = "label"

FEATURE_COLS = [
    "age",
    "bmi",
    "children",
    "claim_amount",
    "days_since_last_claim",
    "charge_per_age",
    "log_claim_amount",
    "smoker_obese_flag",
    "high_value_claim",
    "rapid_resubmission",
    "young_high_procedure_flag",
    "provider_high_volume_flag",
    "claim_above_region_p95",
    "anomaly_score_proxy",
    "provider_claim_deviation",
    "claim_zscore_region",
    "sex_idx",
    "region_idx",
    "bmi_cat_idx",
    "age_group_idx",
    "procedure_risk_idx",
]


def build_session():
    return (
        SparkSession.builder
        .appName("FraudDetectionMLTraining")
        .config("spark.sql.warehouse.dir", "hdfs://hdfs-namenode:9000/user/hive/warehouse")
        .config("spark.sql.shuffle.partitions", "8")
        .enableHiveSupport()
        .getOrCreate()
    )


def load_and_prepare(spark):
    df = spark.read.parquet(HDFS_FEATURES)
    df = df.withColumn(LABEL, col("is_fraud").cast("integer"))
    return df


def oversample_minority(df):
    fraud = df.filter(col("is_fraud") == True)
    normal = df.filter(col("is_fraud") == False)
    n_fraud = fraud.count()
    n_normal = normal.count()
    ratio = n_normal / n_fraud
    capped_ratio = min(ratio, 5.0)
    oversampled = fraud.sample(withReplacement=True, fraction=capped_ratio, seed=42)
    return normal.union(oversampled)


def build_assembler():
    return VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip",
    )


def evaluate_predictions(predictions):
    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL, rawPredictionCol="rawPrediction"
    )
    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL, predictionCol="prediction"
    )

    return {
        "auc_roc": round(binary_eval.setMetricName("areaUnderROC").evaluate(predictions), 4),
        "auc_pr": round(binary_eval.setMetricName("areaUnderPR").evaluate(predictions), 4),
        "f1": round(mc_eval.setMetricName("f1").evaluate(predictions), 4),
        "precision": round(mc_eval.setMetricName("weightedPrecision").evaluate(predictions), 4),
        "recall": round(mc_eval.setMetricName("weightedRecall").evaluate(predictions), 4),
        "accuracy": round(mc_eval.setMetricName("accuracy").evaluate(predictions), 4),
    }


def confusion_matrix_counts(predictions):
    tp = predictions.filter((col(LABEL) == 1) & (col("prediction") == 1.0)).count()
    tn = predictions.filter((col(LABEL) == 0) & (col("prediction") == 0.0)).count()
    fp = predictions.filter((col(LABEL) == 0) & (col("prediction") == 1.0)).count()
    fn = predictions.filter((col(LABEL) == 1) & (col("prediction") == 0.0)).count()
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def train_gbt_with_cv(train_df, assembler):
    gbt = GBTClassifier(
        labelCol=LABEL,
        featuresCol="features",
        maxIter=80,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        featureSubsetStrategy="sqrt",
        seed=42,
    )
    pipeline = Pipeline(stages=[assembler, gbt])

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [60, 80])
        .addGrid(gbt.maxDepth, [4, 5])
        .build()
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC"),
        numFolds=3,
        seed=42,
    )
    print("Running 3-fold CV for GBT (4 param combinations)...")
    cv_model = cv.fit(train_df)
    print(f"Best CV AUC-ROC: {round(max(cv_model.avgMetrics), 4)}")
    return cv_model.bestModel


def train_lr_baseline(train_df, assembler):
    lr = LogisticRegression(
        labelCol=LABEL,
        featuresCol="features",
        maxIter=150,
        regParam=0.01,
        elasticNetParam=0.0,
    )
    pipeline = Pipeline(stages=[assembler, lr])
    return pipeline.fit(train_df)


def save_metrics(gbt_metrics, lr_metrics, cm):
    os.makedirs(METRICS_DIR, exist_ok=True)

    output = {
        "gbt": {**gbt_metrics, "model": "GBTClassifier", "confusion_matrix": cm},
        "logistic_regression": {**lr_metrics, "model": "LogisticRegression"},
    }
    with open(f"{METRICS_DIR}/training_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    rows = [
        ["", "Predicted_Normal", "Predicted_Fraud"],
        ["Actual_Normal", cm["tn"], cm["fp"]],
        ["Actual_Fraud", cm["fn"], cm["tp"]],
    ]
    with open(f"{METRICS_DIR}/confusion_matrix.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Metrics saved to {METRICS_DIR}")


def save_prediction_sample(predictions):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    sample = (
        predictions.select(
            "claim_id", "age", "bmi", "children", "claim_amount",
            "region", "procedure_code", "provider_id",
            "days_since_last_claim", "anomaly_score_proxy",
            LABEL, "prediction", "probability",
        )
        .limit(600)
        .toPandas()
    )
    sample["prob_fraud"] = sample["probability"].apply(lambda v: round(float(v[1]), 6))
    sample.drop(columns=["probability"], inplace=True)
    sample.to_csv(f"{PREDICTIONS_DIR}/predictions_sample.csv", index=False)
    print(f"Predictions sample saved: {PREDICTIONS_DIR}/predictions_sample.csv")


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    print("Loading feature dataset...")
    df = load_and_prepare(spark)
    total = df.count()
    print(f"Total records: {total}")

    balanced = oversample_minority(df)
    train, test = balanced.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train.count()}, Test: {test.count()}")

    assembler = build_assembler()

    print("Training GBT with cross-validation...")
    gbt_model = train_gbt_with_cv(train, assembler)

    print("Training Logistic Regression baseline...")
    lr_model = train_lr_baseline(train, assembler)

    gbt_preds = gbt_model.transform(test)
    lr_preds = lr_model.transform(test)

    gbt_metrics = evaluate_predictions(gbt_preds)
    lr_metrics = evaluate_predictions(lr_preds)
    cm = confusion_matrix_counts(gbt_preds)

    print("\nGBT Metrics:", gbt_metrics)
    print("LR  Metrics:", lr_metrics)
    print("Confusion Matrix:", cm)

    print(f"\nSaving GBT model to {GBT_MODEL_PATH}")
    gbt_model.save(GBT_MODEL_PATH)

    print(f"Saving LR model to {LR_MODEL_PATH}")
    lr_model.save(LR_MODEL_PATH)

    save_metrics(gbt_metrics, lr_metrics, cm)
    save_prediction_sample(gbt_preds)

    spark.stop()
    print("ML training complete.")


if __name__ == "__main__":
    main()
