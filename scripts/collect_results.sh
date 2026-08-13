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
mkdir -p "results/${job}"
kubectl -n "$KUBE_NAMESPACE" cp "$pod:/results/." "results/${job}/"
echo "Saved logs and metrics under results/${job}*"
