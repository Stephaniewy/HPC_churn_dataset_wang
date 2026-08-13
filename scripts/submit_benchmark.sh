#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT:=soe-hpccenter}"
: "${REGISTRY_REGION:=us-central1}"
: "${TEAM:=ways-58}"
: "${RUN_ID:=$(date -u +%Y%m%d-%H%M%S)}"
: "${KUBE_NAMESPACE:=default}"
REPO="${REGISTRY_REGION}-docker.pkg.dev/${PROJECT}/tpu-images"
target="${1:?usage: scripts/submit_benchmark.sh <cpu|gpu|tpu>}"

case "$target" in
  cpu) manifest=infra/cpu-job.yaml; export CPU_IMAGE="${REPO}/churn-cpu-${TEAM}:latest" ;;
  gpu) manifest=infra/gpu-job.yaml; export GPU_IMAGE="${REPO}/churn-gpu-${TEAM}:latest" ;;
  tpu) manifest=infra/tpu-job.yaml; export TPU_IMAGE="${REPO}/churn-tpu-${TEAM}:latest" ;;
  *) echo "target must be cpu, gpu, or tpu" >&2; exit 2 ;;
esac

context=$(kubectl config current-context)
cluster=$(kubectl config view --minify -o jsonpath='{.contexts[0].context.cluster}')
case "$target:$cluster" in
  gpu:stanford-pilot) ;;
  gpu:*) echo "Refusing GPU submission from context '$context' (cluster '$cluster'); expected stanford-pilot cluster" >&2; exit 1 ;;
  tpu:gke_${PROJECT}_us-west4_class-tpu-cluster-west4) ;;
  tpu:*) echo "Refusing TPU submission from context '$context'; expected west4 course GKE context" >&2; exit 1 ;;
  cpu:*) ;;
esac

export RUN_ID TEAM
envsubst < "$manifest" | kubectl -n "$KUBE_NAMESPACE" apply -f -
job="churn-${target}-${TEAM}-${RUN_ID}"
echo "Submitted ${KUBE_NAMESPACE}/${job}"
echo "Collect after completion: KUBE_NAMESPACE=${KUBE_NAMESPACE} scripts/collect_results.sh ${job}"
