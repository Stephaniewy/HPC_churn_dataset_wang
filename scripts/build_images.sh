#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT:=soe-hpccenter}"
: "${REGISTRY_REGION:=us-central1}"
: "${TEAM:=ways-58}"
REPO="${REGISTRY_REGION}-docker.pkg.dev/${PROJECT}/tpu-images"
target="${1:?usage: scripts/build_images.sh <cpu|gpu|tpu>}"

case "$target" in
  cpu)
    command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 1; }
    gcloud builds submit --config infra/cloudbuild-cpu.yaml \
      --substitutions="_IMAGE=${REPO}/churn-cpu-${TEAM}:latest" .
    ;;
  tpu)
    command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 1; }
    gcloud builds submit --config infra/cloudbuild-tpu.yaml \
      --substitutions="_IMAGE=${REPO}/churn-tpu-${TEAM}:latest" .
    ;;
  gpu)
    command -v docker >/dev/null || { echo "docker with buildx is required" >&2; exit 1; }
    echo "Building a no-RUN ARM64 overlay on the NVIDIA JAX image for GH200."
    docker build --platform linux/arm64 -f Dockerfile.gpu \
      -t "${REPO}/churn-gpu-${TEAM}:latest" .
    docker push "${REPO}/churn-gpu-${TEAM}:latest"
    ;;
  *) echo "target must be cpu, gpu, or tpu" >&2; exit 2 ;;
esac
