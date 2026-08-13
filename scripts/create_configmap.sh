#!/usr/bin/env bash
set -euo pipefail
: "${KUBE_NAMESPACE:=default}"
: "${TEAM:=ways-58}"
CONFIGMAP_NAME="churn-data-${TEAM}"
# `kubectl apply` stores a second copy of the CSV in its last-applied annotation,
# which exceeds Kubernetes' 256 KiB annotation limit. Recreate only our uniquely
# team-named ConfigMap and keep the CSV solely in the ConfigMap data field.
kubectl -n "$KUBE_NAMESPACE" delete configmap "$CONFIGMAP_NAME" --ignore-not-found
kubectl -n "$KUBE_NAMESPACE" create configmap "$CONFIGMAP_NAME" \
  --from-file=Churn_Dataset.csv=data/Churn_Dataset.csv
echo "Configured ${KUBE_NAMESPACE}/${CONFIGMAP_NAME}"
