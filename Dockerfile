# Build with one JAX runtime chosen at build time; code and dependencies are immutable.
# Examples: --build-arg JAX_PACKAGE='jax==0.6.2' (CPU)
#           --build-arg JAX_PACKAGE='jax[cuda12]==0.6.2' (GPU)
#           --build-arg JAX_PACKAGE='jax[tpu]==0.6.2' --build-arg JAX_FIND_LINKS=... (TPU)
FROM python:3.11-slim

ARG JAX_PACKAGE="jax==0.6.2"
ARG JAX_FIND_LINKS=""
ARG EXTRA_PACKAGES=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    if [ -n "$JAX_FIND_LINKS" ]; then pip install "$JAX_PACKAGE" -f "$JAX_FIND_LINKS"; else pip install "$JAX_PACKAGE"; fi && \
    if [ -n "$EXTRA_PACKAGES" ]; then pip install $EXTRA_PACKAGES; fi

COPY src/ src/
COPY scripts/container_entrypoint.sh /usr/local/bin/container_entrypoint.sh
RUN chmod +x /usr/local/bin/container_entrypoint.sh

# The dataset is intentionally not copied into this image. Mount it read-only at /input.
ENTRYPOINT ["/usr/local/bin/container_entrypoint.sh"]
CMD ["python", "-u", "src/train.py"]
