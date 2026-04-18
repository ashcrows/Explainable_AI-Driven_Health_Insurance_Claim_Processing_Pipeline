#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KAFKA_DIR="$PROJECT_ROOT/kafka"

RATE="${1:-2}"
FRAUD_RATE="${2:-0.08}"
LIMIT="${3:-0}"

# ── Ensure we are in a virtual environment ────────────────────
if [[ -z "$VIRTUAL_ENV" ]]; then
  echo "ERROR: No virtual environment active."
  echo ""
  echo "Please activate your venv first:"
  echo "  python3 -m venv $PROJECT_ROOT/.venv"
  echo "  source $PROJECT_ROOT/.venv/bin/activate"
  echo "  bash scripts/start_producer.sh"
  exit 1
fi

echo "Installing Kafka producer dependencies into venv..."
pip install --quiet -r "$KAFKA_DIR/requirements.txt"

echo ""
echo "Starting Kafka producer:"
echo "  Bootstrap : localhost:9092"
echo "  Topic     : health_insurance_claims"
echo "  Rate      : ${RATE} events/sec"
echo "  Fraud rate: ${FRAUD_RATE} (${FRAUD_RATE/0./}% fraud)"
echo "  Limit     : ${LIMIT} (0 = unlimited)"
echo ""
echo "Press Ctrl+C to stop the producer."
echo ""

python3 "$KAFKA_DIR/producer.py" \
  --bootstrap-servers localhost:9092 \
  --topic health_insurance_claims \
  --rate "$RATE" \
  --fraud-rate "$FRAUD_RATE" \
  --limit "$LIMIT"
