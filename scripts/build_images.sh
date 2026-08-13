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
    echo "Building the GH200 ARM64 image. Confirm this matches the Lab 1 registry workflow."
    docker buildx build --platform linux/arm64 --push \
      --build-arg 'JAX_PACKAGE=jax[cuda12]==0.6.2' \
      -t "${REPO}/churn-gpu-${TEAM}:latest" .
    ;;
  *) echo "target must be cpu, gpu, or tpu" >&2; exit 2 ;;
esac
