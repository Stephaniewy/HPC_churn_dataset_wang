#!/usr/bin/env bash
set -euo pipefail

: "${TEAM:=ways-58}"
mode="${1:?usage: scripts/run_cpu.sh <verify|benchmark>}"
image="churn-cpu-${TEAM}:local"

command -v docker >/dev/null || {
  echo "Docker is required on the assigned CPU server." >&2
  exit 1
}
docker info >/dev/null
mkdir -p results

case "$mode" in
  verify) epochs=1; run_name="cpu-${TEAM}-docker-check" ;;
  benchmark) epochs=200; run_name="cpu-${TEAM}-$(date -u +%Y%m%d-%H%M%S)" ;;
  *) echo "mode must be verify or benchmark" >&2; exit 2 ;;
esac

docker build --build-arg 'JAX_PACKAGE=jax==0.6.2' -t "$image" .
docker run --rm \
  --cpus "${CPU_COUNT:-4}" --memory "${CPU_MEMORY:-8g}" \
  -v "$PWD/data:/input:ro" \
  -v "$PWD/results:/workspace/results" \
  "$image" --run-name "$run_name" --epochs "$epochs" \
  | tee "results/${run_name}.log"

echo "Saved results/${run_name}.json, results/${run_name}.log, and results/benchmark_summary.csv"
