#!/usr/bin/env bash
# Usage: scripts/smoke_test.sh [base_url]   (default http://localhost:8000)
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"

echo "Waiting for ${BASE_URL}/health ..."
for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then break; fi
  sleep 2
done

echo "GET /health";  curl -fsS "${BASE_URL}/health"; echo
echo "POST /predict"
RESP=$(curl -fsS -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d @"$(dirname "$0")/../sample_input.json")
echo "${RESP}"
echo "${RESP}" | grep -q '"confidence"' || { echo "confidence missing"; exit 1; }
echo "GET /metrics (first lines)"; curl -fsS "${BASE_URL}/metrics" | grep -m3 api_
echo "Smoke test passed."
