import os
import json
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

HDFS_FEATURES = "hdfs://hdfs-namenode:9000/user/health/features"
METRICS_DIR = "/opt/spark-apps/outputs/metrics"
PREDICTIONS_DIR = "/opt/spark-apps/outputs/predictions"
PLOTS_DIR = "/opt/spark-apps/outputs/plots"
SAMPLE_SIZE = 8000

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

LABEL = "is_fraud"


def build_session():
    return (
        SparkSession.builder
        .appName("SHAPExplainabilityLayer")
        .config("spark.sql.warehouse.dir", "hdfs://hdfs-namenode:9000/user/hive/warehouse")
        .enableHiveSupport()
        .getOrCreate()
    )


def collect_sample(spark):
    df = spark.read.parquet(HDFS_FEATURES)
    fraud_sample = df.filter(col(LABEL) == True).limit(SAMPLE_SIZE // 2)
    normal_sample = df.filter(col(LABEL) == False).limit(SAMPLE_SIZE // 2)
    combined = fraud_sample.union(normal_sample)
    pdf = combined.select(FEATURE_COLS + [LABEL]).toPandas()
    pdf[LABEL] = pdf[LABEL].astype(int)
    pdf[FEATURE_COLS] = pdf[FEATURE_COLS].fillna(0.0)
    return pdf


def train_local_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value


def global_importance(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = {feat: round(float(mean_abs[i]), 6) for i, feat in enumerate(feature_names)}
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def save_global_importance(importance_dict):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = f"{METRICS_DIR}/shap_global_importance.json"
    with open(path, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Global SHAP importance saved: {path}")


def save_shap_sample_csv(shap_values, X, y, probs):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{c}" for c in FEATURE_COLS],
    )
    shap_df["true_label"] = y.values
    shap_df["predicted_fraud_prob"] = probs
    for fc in FEATURE_COLS:
        shap_df[f"feat_{fc}"] = X[fc].values

    shap_df.head(250).to_csv(f"{PREDICTIONS_DIR}/shap_values_sample.csv", index=False)
    print(f"SHAP sample CSV saved: {PREDICTIONS_DIR}/shap_values_sample.csv")


def save_plots(shap_values, X, model, importance_dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Beeswarm plot saved: {PLOTS_DIR}/shap_beeswarm.png")

        shap.summary_plot(
            shap_values, X, feature_names=FEATURE_COLS,
            plot_type="bar", show=False, max_display=15,
        )
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_feature_importance_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Bar importance plot saved: {PLOTS_DIR}/shap_feature_importance_bar.png")

        top_feat = list(importance_dict.keys())[0]
        feat_idx = FEATURE_COLS.index(top_feat)
        shap.dependence_plot(feat_idx, shap_values, X, feature_names=FEATURE_COLS, show=False)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_dependence_{top_feat}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Dependence plot saved for: {top_feat}")

    except Exception as exc:
        print(f"Plot generation skipped ({exc}). Saving placeholder notes instead.")
        for name in ["shap_beeswarm", "shap_feature_importance_bar"]:
            with open(f"{PLOTS_DIR}/{name}_placeholder.txt", "w") as f:
                f.write(
                    f"Plot '{name}' would be generated during a full local run with matplotlib.\n"
                    f"Top features by SHAP importance:\n"
                )
                for feat, val in list(importance_dict.items())[:10]:
                    f.write(f"  {feat}: {val}\n")


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Collecting {SAMPLE_SIZE} samples from HDFS features...")
    pdf = collect_sample(spark)
    spark.stop()

    X = pdf[FEATURE_COLS]
    y = pdf[LABEL]
    print(f"Collected: {len(X)} rows, fraud rate: {round(y.mean()*100,2)}%")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training local sklearn GBT for SHAP explanation...")
    model = train_local_model(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Local GBT accuracy - train: {round(train_acc,4)}, test: {round(test_acc,4)}")

    print("Computing SHAP values...")
    shap_values, base_value = compute_shap_values(model, X)

    importance = global_importance(shap_values, FEATURE_COLS)

    print("\nTop-10 features by mean absolute SHAP:")
    for feat, val in list(importance.items())[:10]:
        print(f"  {feat:35s}: {val:.6f}")

    save_global_importance(importance)

    probs = model.predict_proba(X)[:, 1]
    save_shap_sample_csv(shap_values, X, y, probs)

    save_plots(shap_values, X, model, importance)

    print("\nSHAP explainability complete.")


if __name__ == "__main__":
    main()
