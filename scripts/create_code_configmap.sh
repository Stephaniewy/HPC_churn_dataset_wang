#!/usr/bin/env bash
set -euo pipefail
: "${KUBE_NAMESPACE:=default}"
: "${TEAM:=ways-58}"
name="churn-code-${TEAM}"

kubectl -n "$KUBE_NAMESPACE" delete configmap "$name" --ignore-not-found
kubectl -n "$KUBE_NAMESPACE" create configmap "$name" --from-file=train.py=src/train.py
echo "Configured ${KUBE_NAMESPACE}/${name}"
