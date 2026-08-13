#!/usr/bin/env bash
set -euo pipefail
: "${KUBE_NAMESPACE:=default}"
: "${TEAM:=ways-58}"
CONFIGMAP_NAME="churn-data-${TEAM}"
kubectl -n "$KUBE_NAMESPACE" create configmap "$CONFIGMAP_NAME" \
  --from-file=Churn_Dataset.csv=data/Churn_Dataset.csv \
  --dry-run=client -o yaml | kubectl -n "$KUBE_NAMESPACE" apply -f -
echo "Configured ${KUBE_NAMESPACE}/${CONFIGMAP_NAME}"
