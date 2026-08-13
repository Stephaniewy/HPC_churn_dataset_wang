#!/usr/bin/env bash
set -euo pipefail
: "${KUBE_NAMESPACE:=default}"
job="${1:?usage: scripts/collect_results.sh <job-name>}"

mkdir -p results
echo "Waiting for ${KUBE_NAMESPACE}/${job} (timeout: ${WAIT_TIMEOUT:-60m})"
if ! kubectl -n "$KUBE_NAMESPACE" wait --for=condition=complete "job/$job" --timeout="${WAIT_TIMEOUT:-60m}"; then
  kubectl -n "$KUBE_NAMESPACE" describe "job/$job" > "results/${job}.describe.log" || true
  kubectl -n "$KUBE_NAMESPACE" get pods -l "job-name=$job" -o wide || true
  exit 1
fi
pod=$(kubectl -n "$KUBE_NAMESPACE" get pods -l "job-name=$job" -o jsonpath='{.items[0].metadata.name}')
kubectl -n "$KUBE_NAMESPACE" logs "$pod" > "results/${job}.log"
sed -n '/^-----BEGIN BENCHMARK JSON-----$/,/^-----END BENCHMARK JSON-----$/p' "results/${job}.log" \
  | sed '1d;$d' > "results/${job}.json"
sed -n '/^-----BEGIN BENCHMARK CSV-----$/,/^-----END BENCHMARK CSV-----$/p' "results/${job}.log" \
  | sed '1d;$d' > "results/${job}.csv"

python3 -m json.tool "results/${job}.json" >/dev/null
if [ ! -s results/benchmark_summary.csv ]; then
  cp "results/${job}.csv" results/benchmark_summary.csv
else
  tail -n +2 "results/${job}.csv" >> results/benchmark_summary.csv
fi
echo "Saved results/${job}.log, .json, .csv and updated benchmark_summary.csv"
