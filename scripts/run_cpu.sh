#!/usr/bin/env bash
set -euo pipefail

: "${TEAM:=ways-58}"
mode="${1:?usage: scripts/run_cpu.sh <verify|benchmark>}"
image="churn-cpu-${TEAM}:local"

command -v docker >/dev/null || {
  echo "Docker is required on the assigned CPU server." >&2
  exit 1
}
docker_details=$(docker info 2>&1)
mkdir -p results

cpu_count="${CPU_COUNT:-4}"
cpu_memory="${CPU_MEMORY:-8g}"
cpu_args=()
data_volume_mode="ro"
results_volume_mode=""

# The assigned Rocky Linux server exposes rootless Podman without CPU/cpuset
# cgroup controllers. Run on the assigned node and record the visible CPUs and
# actual utilization in telemetry. Native Docker can enforce an explicit limit.
if grep -qi podman <<<"$docker_details"; then
  data_volume_mode="ro,Z"
  results_volume_mode=":Z"
  echo "Rootless Podman detected: using the assigned CPU node; telemetry records visible CPUs and utilization."
else
  cpu_args=(--cpus "$cpu_count")
fi

case "$mode" in
  verify) epochs=1; run_name="cpu-${TEAM}-docker-check" ;;
  benchmark) epochs=200; run_name="cpu-${TEAM}-$(date -u +%Y%m%d-%H%M%S)" ;;
  *) echo "mode must be verify or benchmark" >&2; exit 2 ;;
esac

docker build --platform linux/amd64 \
  --build-arg 'JAX_PACKAGE=jax==0.6.2' -t "$image" .
docker run --rm \
  "${cpu_args[@]}" --memory "$cpu_memory" \
  -v "$PWD/data:/input:${data_volume_mode}" \
  -v "$PWD/results:/workspace/results${results_volume_mode}" \
  "$image" --run-name "$run_name" --epochs "$epochs" \
  | tee "results/${run_name}.log"

echo "Saved results/${run_name}.json, results/${run_name}.log, and results/benchmark_summary.csv"
