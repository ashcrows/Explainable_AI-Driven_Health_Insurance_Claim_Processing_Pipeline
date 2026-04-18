#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"

echo "========================================"
echo " Health Insurance XAI Pipeline — Start"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ── macOS pre-flight checks ──────────────────────────────────
echo "[1/7] Pre-flight checks..."

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Docker Desktop is not running. Please start it first."
  exit 1
fi

# Check Rosetta 2 (needed for bde2020 x86 images on Apple Silicon)
if [[ "$(uname -m)" == "arm64" ]]; then
  if ! /usr/bin/pgrep -q oahd 2>/dev/null && ! arch -x86_64 true 2>/dev/null; then
    echo "WARNING: Rosetta 2 may not be installed."
    echo "Run: softwareupdate --install-rosetta --agree-to-license"
    echo "Continuing anyway — Docker will attempt emulation..."
  fi
  echo "Apple Silicon detected. bde2020 Hadoop/Hive images run via Rosetta 2 (expect slower startup)."
fi

# Check Docker memory allocation
DOCKER_MEM_BYTES=$(docker system info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
DOCKER_MEM_GB=$(echo "$DOCKER_MEM_BYTES / 1073741824" | bc 2>/dev/null || echo "0")
if [[ "$DOCKER_MEM_GB" -lt 9 ]] 2>/dev/null; then
  echo ""
  echo "WARNING: Docker has only ~${DOCKER_MEM_GB}GB RAM allocated."
  echo "This stack needs at least 10GB. Go to:"
  echo "  Docker Desktop → Settings → Resources → Memory → 10GB+"
  echo ""
fi

# Check if AirPlay Receiver is on port 9000 (we use 19000 instead, just informational)
echo "Note: HDFS NameNode RPC is mapped to host port 19000 (not 9000) to avoid AirPlay conflict."
echo ""

# ── Start services in dependency order ──────────────────────
cd "$DOCKER_DIR"

echo "[2/7] Starting Zookeeper and PostgreSQL..."
docker compose up -d zookeeper postgres
echo "Waiting 20s for Zookeeper and PostgreSQL to be healthy..."
sleep 20

echo ""
echo "[3/7] Starting Kafka and HDFS..."
docker compose up -d kafka hdfs-namenode hdfs-datanode
echo "Waiting 30s for Kafka and HDFS (Hadoop images emulate on Apple Silicon — may take longer)..."
sleep 30

echo ""
echo "[4/7] Starting Hive Metastore..."
docker compose up -d hive-metastore
echo "Waiting 35s for Hive Metastore schema init..."
sleep 35

echo ""
echo "[5/7] Starting Spark cluster (master + 2 workers)..."
echo "Note: Spark workers will pip-install shap + sklearn on first boot (~2-3 min)..."
docker compose up -d spark-master spark-worker-1 spark-worker-2
echo "Waiting 180s for Spark workers to install Python dependencies..."
sleep 180

echo ""
echo "[6/7] Initializing Airflow..."
docker compose up -d airflow-init
echo "Waiting 30s for Airflow DB init..."
sleep 30
docker compose up -d airflow-webserver airflow-scheduler
echo "Waiting 15s for Airflow webserver..."
sleep 15

echo ""
echo "[7/7] Starting Grafana and Superset..."
docker compose up -d grafana superset
echo "Waiting 20s for Grafana and Superset init..."
sleep 20

# ── HDFS directory setup ─────────────────────────────────────
echo ""
echo "Creating HDFS directory structure..."
docker exec hdfs-namenode hdfs dfs -mkdir -p /user/health/raw_claims
docker exec hdfs-namenode hdfs dfs -mkdir -p /user/health/features
docker exec hdfs-namenode hdfs dfs -mkdir -p /user/health/models
docker exec hdfs-namenode hdfs dfs -mkdir -p /user/health/checkpoints/streaming_v1
docker exec hdfs-namenode hdfs dfs -mkdir -p /user/hive/warehouse
docker exec hdfs-namenode hdfs dfs -chmod -R 777 /user
echo "HDFS directories created."

# ── Kafka topic ──────────────────────────────────────────────
echo ""
echo "Creating Kafka topic..."
docker exec kafka kafka-topics --create \
  --bootstrap-server kafka:29092 \
  --topic health_insurance_claims \
  --partitions 1 \
  --replication-factor 1 \
  --if-not-exists
echo "Kafka topic ready."

# ── Verify Spark workers have shap installed ─────────────────
echo ""
echo "Verifying shap installation on Spark workers..."
docker exec spark-worker-1 python3 -c "import shap; print('spark-worker-1: shap OK')" 2>/dev/null || \
  echo "spark-worker-1: shap still installing — check with: docker logs spark-worker-1"
docker exec spark-worker-2 python3 -c "import shap; print('spark-worker-2: shap OK')" 2>/dev/null || \
  echo "spark-worker-2: shap still installing — check with: docker logs spark-worker-2"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo " All services started successfully."
echo "========================================"
echo ""
echo "  Spark UI          : http://localhost:8080"
echo "  Spark Worker 1    : http://localhost:8081"
echo "  Spark Worker 2    : http://localhost:8082"
echo "  HDFS NameNode UI  : http://localhost:9870"
echo "  Airflow UI        : http://localhost:8083  (admin / admin123)"
echo "  Grafana           : http://localhost:3000  (admin / grafana123)"
echo "  Superset          : http://localhost:8088  (admin / admin123)"
echo "  PostgreSQL        : localhost:5432"
echo "  Kafka             : localhost:9092"
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_ROOT"
echo "  2. python3 -m venv .venv && source .venv/bin/activate"
echo "  3. bash scripts/start_producer.sh        # stream synthetic claims"
echo "  4. bash scripts/submit_streaming.sh      # ingest into HDFS"
echo "  5. Wait 5-10 min, then:"
echo "  6. bash scripts/submit_batch.sh          # feature eng + ML + SHAP"
echo ""
