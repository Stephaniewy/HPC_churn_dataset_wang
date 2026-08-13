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
python3 - "results/${job}.csv" results/benchmark_summary.csv <<'PY'
import csv
import sys
from pathlib import Path

run_path, summary_path = map(Path, sys.argv[1:])
with run_path.open(newline="") as f:
    run_row = next(csv.DictReader(f))
prior = []
fields = list(run_row)
if summary_path.exists() and summary_path.stat().st_size:
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        prior = [row for row in reader if row.get("run_name") != run_row["run_name"]]
        for field in reader.fieldnames or []:
            if field not in fields:
                fields.append(field)
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(prior)
    writer.writerow(run_row)
PY
echo "Saved results/${job}.log, .json, .csv and updated benchmark_summary.csv"
