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
run_prefix=()
cpu_args=()
volume_label=""

# The assigned Rocky Linux server exposes Docker-compatible Podman without the
# rootless CPU cgroup controller. Pin the runtime process instead; the container
# inherits that CPU affinity. Native Docker can use its normal --cpus limit.
if grep -qi podman <<<"$docker_details"; then
  command -v taskset >/dev/null || {
    echo "Podman without taskset cannot enforce a reproducible CPU allocation." >&2
    exit 1
  }
  cpu_set="${CPUSET:-0-$((cpu_count - 1))}"
  if ! taskset -c "$cpu_set" true 2>/dev/null; then
    echo "CPUSET '$cpu_set' is unavailable; allowed CPUs: $(taskset -pc $$ 2>/dev/null || echo unknown)" >&2
    exit 1
  fi
  run_prefix=(taskset -c "$cpu_set")
  volume_label=",Z"
  echo "Podman detected: pinning the container to CPUSET=${cpu_set}."
else
  cpu_args=(--cpus "$cpu_count")
fi

case "$mode" in
  verify) epochs=1; run_name="cpu-${TEAM}-docker-check" ;;
  benchmark) epochs=200; run_name="cpu-${TEAM}-$(date -u +%Y%m%d-%H%M%S)" ;;
  *) echo "mode must be verify or benchmark" >&2; exit 2 ;;
esac

docker build --build-arg 'JAX_PACKAGE=jax==0.6.2' -t "$image" .
"${run_prefix[@]}" docker run --rm \
  "${cpu_args[@]}" --memory "$cpu_memory" \
  -v "$PWD/data:/input:ro${volume_label}" \
  -v "$PWD/results:/workspace/results${volume_label}" \
  "$image" --run-name "$run_name" --epochs "$epochs" \
  | tee "results/${run_name}.log"

echo "Saved results/${run_name}.json, results/${run_name}.log, and results/benchmark_summary.csv"
