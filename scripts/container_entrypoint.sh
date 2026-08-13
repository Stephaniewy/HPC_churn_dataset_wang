#!/usr/bin/env sh
set -eu

# Use node-local RAM after a read-only volume mount, preventing network/storage input stalls.
if [ -f /input/Churn_Dataset.csv ]; then
  mkdir -p /dev/shm/churn
  cp /input/Churn_Dataset.csv /dev/shm/churn/Churn_Dataset.csv
  export DATA_PATH=/dev/shm/churn/Churn_Dataset.csv
fi

# Docker/Kubernetes arguments beginning with an option are training arguments.
# This preserves the image's default command when a Job supplies only `args`.
if [ "${1#-}" != "$1" ]; then
  set -- python -u src/train.py "$@"
fi

exec "$@"
