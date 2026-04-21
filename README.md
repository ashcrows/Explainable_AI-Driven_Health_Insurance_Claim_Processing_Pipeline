# Health Insurance XAI Pipeline

End-to-end fraud detection pipeline for health insurance claims using Kafka streaming, Apache Spark MLlib, SHAP explainability, and Airflow orchestration.

---

**Course:** CSL7110 - Machine Learning with Big Data  
**Team:** Ashish Sinha (M25DE1047), Aniket Srivastava (M25DE1051), Akshay Kumar (M25DE1028)  
**Runtime:** Local execution using Docker Compose

---

## Architecture

```
Kafka Producer (host)
      │ synthetic claim events (2/sec)
      ▼
Kafka Broker ──► Spark Structured Streaming ──► HDFS (raw_claims/)
                                                      │
                                              Spark Batch ETL
                                              (feature engineering)
                                                      │
                                              HDFS (features/) + Hive table
                                                      │
                                         ┌────────────┴────────────┐
                                   GBT Classifier          Logistic Regression
                                   (3-fold CV)              (baseline)
                                         │
                                   SHAP Explainability
                                   (sklearn TreeExplainer)
                                         │
                                   SQL Analytics (Hive)
                                         │
                              Grafana / Superset dashboards
                                         │
                              Airflow DAG (weekly retraining)
```

### Services and ports

| Service           | Host Port | Container Port | Credentials         |
|-------------------|-----------|----------------|---------------------|
| Spark Master UI   | 8080      | 8080           | —                   |
| Spark Worker 1 UI | 8081      | 8081           | —                   |
| Spark Worker 2 UI | 8082      | 8081           | —                   |
| HDFS NameNode UI  | 9870      | 9870           | —                   |
| HDFS RPC (host)   | **19000** | 9000           | remapped (AirPlay)  |
| Airflow           | 8083      | 8080           | admin / admin123    |
| Grafana           | 3000      | 3000           | admin / grafana123  |
| Superset          | 8088      | 8088           | admin / admin123    |
| PostgreSQL        | 5432      | 5432           | postgres / postgres123 |
| Kafka             | 9092      | 9092           | —                   |
| Zookeeper         | 2181      | 2181           | —                   |

> **Why port 19000?** macOS Monterey+ reserves port 9000 for AirPlay Receiver. HDFS NameNode RPC is remapped to 19000 on the host. All Spark jobs run inside Docker and continue to use `hdfs://hdfs-namenode:9000` (the internal network port).

---

## Prerequisites

### Required software

| Software | Version | Install |
|----------|---------|---------|
| macOS | 12+ (Monterey or later) | — |
| Docker Desktop | 4.20+ | https://www.docker.com/products/docker-desktop/ |
| Python | 3.9 – 3.11 | `brew install python@3.11` |
| Rosetta 2 | latest | `softwareupdate --install-rosetta --agree-to-license` |

> Rosetta 2 is required because the `bde2020/hadoop-*` and `bde2020/hive` images have no ARM64 build. Docker Desktop uses Rosetta to emulate x86 on Apple Silicon automatically once it is installed.

### Docker Desktop resource settings

Open Docker Desktop → **Settings → Resources** and configure:

| Setting | Minimum | Recommended |
|---------|---------|-------------|
| Memory  | 10 GB   | 12 GB       |
| CPUs    | 4       | 6           |
| Disk    | 40 GB   | 60 GB       |

Click **Apply & Restart** after changing these values.

### Disable AirPlay Receiver (one-time)

```
System Settings → General → AirDrop & Handoff → AirPlay Receiver → OFF
```

This frees port 9000. (The project already remaps HDFS to 19000 as a belt-and-suspenders fix.)

---

## Project structure

```
health-insurance-xai-pipeline/
├── docker/
│   ├── docker-compose.yml        ← main orchestration file
│   ├── hadoop.env                ← HDFS/YARN environment variables
│   ├── init-postgres.sql         ← creates hive + airflow databases
│   ├── hive-site.xml             ← Hive metastore config
│   └── grafana/
│       └── provisioning/
│           ├── datasources/datasource.yml
│           └── dashboards/dashboard.yml
├── kafka/
│   ├── producer.py               ← synthetic claim event generator
│   └── requirements.txt          ← confluent-kafka, faker
├── spark/
│   ├── streaming_ingestion.py    ← Kafka → HDFS (Structured Streaming)
│   ├── batch_feature_engineering.py  ← ETL + 21 ML features
│   ├── ml_training.py            ← GBT + LR with cross-validation
│   ├── shap_explainability.py    ← sklearn GBT + SHAP TreeExplainer
│   ├── sql_analytics.py          ← 7 Hive SQL fraud queries
│   ├── benchmarking.py           ← throughput/latency benchmarks
│   └── requirements.txt          ← installed inside Spark containers
├── airflow/
│   └── dags/
│       └── retraining_dag.py     ← weekly retraining DAG
├── scripts/
│   ├── start_pipeline.sh         ← step 1: boot all services
│   ├── start_producer.sh         ← step 2: run Kafka producer
│   ├── submit_streaming.sh       ← step 3: start stream ingestion
│   └── submit_batch.sh           ← step 4: run full batch pipeline
├── data/
│   └── insurance_sample.csv      ← small reference dataset (10 rows)
├── outputs/                      ← auto-created by Spark jobs
│   ├── metrics/
│   ├── predictions/
│   ├── plots/
│   ├── analytics/
│   └── benchmarks/
└── dashboard/
    ├── grafana_dashboard.json
    └── superset_setup.md
```

---

## Step-by-step execution

### Step 0 — Clone / extract the project

```bash
# If you received a zip file:
unzip health-insurance-xai-pipeline.zip
cd health-insurance-xai-pipeline
```

Make all scripts executable:

```bash
chmod +x scripts/*.sh
```

### Step 1 — Create Python virtual environment

The Kafka producer runs on your Mac (not in Docker). Use a venv to keep dependencies isolated.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your shell prompt. Keep this terminal open — you will need the venv active for Step 4.

### Step 2 — Start all Docker services

```bash
bash scripts/start_pipeline.sh
```

This script:
1. Runs macOS pre-flight checks (Docker running, Rosetta installed, memory available)
2. Starts services in correct dependency order with appropriate wait times
3. Creates HDFS directories
4. Creates the Kafka topic
5. Verifies that SHAP/sklearn are installed on Spark workers

**Expected duration: 4–6 minutes** on first run (Hadoop images are slow on Rosetta). Subsequent runs are faster.

**Verify all services are up:**

```bash
docker compose -f docker/docker-compose.yml ps
```

All services should show `Up` or `running`. If any show `Exit`, check logs:

```bash
docker logs <container_name>
```

**Verify HDFS is ready:**

```bash
docker exec hdfs-namenode hdfs dfs -ls /user/
```

Expected output: directories `health/` and `hive/`.

**Verify Spark workers have shap installed:**

```bash
docker exec spark-worker-1 python3 -c "import shap; print('OK')"
docker exec spark-worker-2 python3 -c "import shap; print('OK')"
```

If either fails, the pip install is still running. Wait 2 minutes and retry.

### Step 3 — Start the Kafka producer (Terminal 1)

Open a terminal, activate the venv, and start producing synthetic claim events:

```bash
source .venv/bin/activate
bash scripts/start_producer.sh
```

Default: 2 events/sec with 8% fraud rate. Override with arguments:

```bash
# Custom rate: 5 events/sec, 10% fraud, unlimited
bash scripts/start_producer.sh 5 0.10 0

# Produce exactly 5000 events and stop
bash scripts/start_producer.sh 2 0.08 5000
```

You will see output like:

```
Producer started: topic=health_insurance_claims rate=2.0/s fraud_rate=0.08
[2026-04-18T10:01:00] Produced 100 events
[2026-04-18T10:02:00] Produced 200 events
```

**Leave this terminal running.**

### Step 4 — Start Spark Structured Streaming (Terminal 2)

Open a new terminal and submit the streaming ingestion job:

```bash
bash scripts/submit_streaming.sh
```

This job reads from Kafka and writes Parquet files to HDFS every 30 seconds, partitioned by `region`.

**Verify data is flowing into HDFS** (after ~1 minute):

```bash
docker exec hdfs-namenode hdfs dfs -ls /user/health/raw_claims/
```

You should see `region=northeast/`, `region=northwest/`, etc.

**Count ingested records:**

```bash
docker exec hdfs-namenode hdfs dfs -count /user/health/raw_claims/
```

The second column shows file count. Wait until you have at least a few hundred records before running batch jobs. For meaningful ML training, wait **5–10 minutes** (600–1200 events at the default 2/sec rate).

**Leave this terminal running.**

### Step 5 — Run the batch pipeline (Terminal 3)

Open a third terminal and run the full batch pipeline:

```bash
bash scripts/submit_batch.sh
```

This executes five Spark jobs in sequence:

| Step | Job | Duration |
|------|-----|----------|
| 1 | `batch_feature_engineering.py` | 1–3 min |
| 2 | `ml_training.py` (GBT 3-fold CV + LR) | 5–15 min |
| 3 | `shap_explainability.py` | 2–5 min |
| 4 | `sql_analytics.py` | 1–2 min |
| 5 | `benchmarking.py` | 2–4 min |

**Total: 11–29 minutes** depending on data volume.

Progress is visible in the Spark UI at http://localhost:8080.

### Step 6 — Trigger the Airflow DAG

Open http://localhost:8083 in a browser.

- Username: `admin`
- Password: `admin123`

1. Click **DAGs** in the top menu
2. Find `health_insurance_weekly_retraining`
3. Toggle it **ON** (the slider on the left)
4. Click the **▶ Run** button (Trigger DAG) on the right
5. Click the DAG name → **Grid** to watch task execution

The DAG runs the same 5-step batch pipeline automatically. It is scheduled weekly (Mondays at 02:00 UTC) and can be triggered manually at any time.

### Step 7 — View results

**Training metrics:**

```bash
cat outputs/metrics/training_metrics.json
```

**SHAP feature importance:**

```bash
cat outputs/metrics/shap_global_importance.json
```

**Predictions sample:**

```bash
head -5 outputs/predictions/predictions_sample.csv
```

**Benchmark results:**

```bash
cat outputs/benchmarks/benchmark_results.csv
```

**SHAP plots** (PNG files):

```bash
open outputs/plots/shap_beeswarm.png
open outputs/plots/shap_feature_importance_bar.png
```

### Step 8 — Grafana dashboard

Open http://localhost:3000

- Username: `admin`
- Password: `grafana123`

The dashboard is pre-provisioned from `docker/grafana/provisioning/`. It connects to PostgreSQL and displays claim metrics.

### Step 9 — Superset (optional)

Open http://localhost:8088

- Username: `admin`
- Password: `admin123`

To connect Superset to the PostgreSQL database:

1. **Settings → Database Connections → + Database**
2. Select **PostgreSQL**
3. Connection string: `postgresql://postgres:postgres123@postgres:5432/postgres`
4. Click **Test Connection** → **Connect**

---

## Stopping the pipeline

### Stop the Kafka producer
In Terminal 1: `Ctrl+C`

### Stop the streaming job
In Terminal 2: `Ctrl+C`

### Stop all Docker containers
```bash
cd docker
docker compose down
```

### Stop and remove all data volumes (full reset)
```bash
cd docker
docker compose down -v
```

> **Warning:** `-v` deletes all HDFS data, Kafka offsets, PostgreSQL databases, and Airflow metadata. Only use this for a clean restart.

---

## Troubleshooting

### Docker containers exit immediately

**Symptom:** `docker compose ps` shows `Exit 137` or `Exit 1`

**Cause:** Out of memory.

**Fix:** Increase Docker Desktop memory to 12 GB (Settings → Resources → Memory).

---

### HDFS NameNode stays in safe mode

**Symptom:** `hdfs dfs -mkdir` returns `Name node is in safe mode`

**Fix:**
```bash
docker exec hdfs-namenode hdfs dfsadmin -safemode leave
```

---

### Spark workers still show "shap not found" after 3 minutes

**Symptom:** `docker exec spark-worker-1 python3 -c "import shap"` raises `ModuleNotFoundError`

**Fix:** The pip install is still running. Check progress:
```bash
docker logs spark-worker-1 --tail 20
```

Wait until you see `Successfully installed shap-0.44.0`. Then retry.

If it failed, reinstall manually:
```bash
docker exec spark-worker-1 pip install shap==0.44.0 scikit-learn==1.3.2 matplotlib==3.8.2
docker exec spark-worker-2 pip install shap==0.44.0 scikit-learn==1.3.2 matplotlib==3.8.2
```

---

### Hive metastore connection refused

**Symptom:** Spark jobs fail with `Could not connect to metastore: thrift://hive-metastore:9083`

**Cause:** Hive metastore is still initializing (it's slow on Rosetta emulation).

**Fix:** Wait 2–3 more minutes, then retry. Check status:
```bash
docker logs hive-metastore --tail 30
```

Look for `Starting Hive Metastore Server` in the output.

---

### Kafka producer: "Failed to connect to localhost:9092"

**Symptom:** Producer exits with `KafkaException: Failed to resolve 'localhost:9092'`

**Cause:** Kafka container is not yet ready, or the `start_pipeline.sh` script was not run first.

**Fix:**
```bash
# Check Kafka is running
docker ps | grep kafka

# Test connectivity
docker exec kafka kafka-topics --list --bootstrap-server kafka:29092
```

---

### SHAP explainability fails with "not enough data"

**Symptom:** `shap_explainability.py` fails because HDFS features are empty or too small.

**Cause:** The streaming job has not ingested enough data yet.

**Fix:** Wait longer (or increase producer rate):
```bash
# Check how many records are in HDFS
docker exec hdfs-namenode hdfs dfs -count /user/health/raw_claims/

# Increase producer rate to 10 events/sec
bash scripts/start_producer.sh 10 0.08 0
```

Run `submit_batch.sh` again once you have at least 500 records.

---

### Port conflict errors

**Symptom:** `docker compose up` fails with `Bind for 0.0.0.0:XXXX failed: port is already allocated`

**Fix:** Find and stop the conflicting process:
```bash
# Find what's using the port (e.g., 8080)
lsof -i :8080

# Kill by PID
kill -9 <PID>
```

Common conflicts on macOS:
- Port 8080: other web servers, Control Center
- Port 5432: local PostgreSQL (`brew services stop postgresql`)
- Port 3000: local Node dev servers

---

### Superset "Internal Server Error" on first load

**Cause:** Superset's database init is still running.

**Fix:** Wait 60 seconds and refresh. Check logs:
```bash
docker logs superset --tail 30
```

---

## Re-running after a restart

If you stopped the containers and want to run again:

```bash
# Start all services (no need to recreate — volumes persist)
cd docker
docker compose up -d

# Wait ~2 minutes for services to fully start, then:
source .venv/bin/activate
bash scripts/start_producer.sh &        # background
bash scripts/submit_streaming.sh &      # background
sleep 300                               # wait 5 min
bash scripts/submit_batch.sh
```

---

## Resource summary

| Container | Memory limit | Image type |
|-----------|-------------|------------|
| zookeeper | 512 MB | ARM64 native |
| kafka | 1 GB | ARM64 native |
| hdfs-namenode | 1 GB | x86 (Rosetta) |
| hdfs-datanode | 1 GB | x86 (Rosetta) |
| postgres | 512 MB | ARM64 native |
| hive-metastore | 1 GB | x86 (Rosetta) |
| spark-master | 1 GB | ARM64 native |
| spark-worker-1 | 2 GB | ARM64 native |
| spark-worker-2 | 1 GB | ARM64 native |
| airflow-webserver | 1 GB | ARM64 native |
| airflow-scheduler | 1 GB | ARM64 native |
| grafana | 256 MB | ARM64 native |
| superset | 1 GB | ARM64 native |
| **Total** | **~12.25 GB** | |

Docker Desktop should have at least **10 GB** allocated. **12 GB** is recommended.

---

## Technology stack

| Layer | Technology |
|-------|-----------|
| Streaming ingest | Apache Kafka 7.4, Spark Structured Streaming 3.4.1 |
| Storage | HDFS 3.2.1, Apache Hive 2.3.2 (PostgreSQL metastore) |
| Feature engineering | PySpark MLlib, Spark SQL |
| ML training | PySpark GBTClassifier, LogisticRegression (3-fold CV) |
| Explainability | SHAP 0.44 TreeExplainer, scikit-learn 1.3.2 GBT |
| Orchestration | Apache Airflow 2.7.2 (LocalExecutor) |
| Dashboards | Grafana 10.1.2, Apache Superset 3.0.0 |
| Database | PostgreSQL 14 |
| Containerization | Docker Compose 3.8 |
