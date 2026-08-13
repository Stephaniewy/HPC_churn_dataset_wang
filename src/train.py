"""Reproducible JAX/XLA DNN churn benchmark for CPU, GH200 GPU, and TPU v5e."""
import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("DATA_PATH", "/input/Churn_Dataset.csv"))
    p.add_argument("--output-dir", default="results")
    p.add_argument("--run-name", default=os.environ.get("RUN_NAME", "benchmark"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=344)
    return p.parse_args()


def load_data(path, seed):
    df = pd.read_csv(path)
    y = df.pop("churned").astype(str).str.upper().eq("TRUE").astype(np.float32).to_numpy()
    X = df.apply(pd.to_numeric, errors="coerce").fillna(df.median(numeric_only=True)).to_numpy(np.float32)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.50, random_state=seed, stratify=y_holdout)
    scaler = StandardScaler().fit(X_train)
    return *(scaler.transform(a).astype(np.float32) for a in (X_train, X_val, X_test)), y_train, y_val, y_test, len(df), X.shape[1]


def init_params(key, input_dim):
    sizes = [input_dim, 64, 32, 1]
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b), dtype=jnp.float32) * jnp.sqrt(2.0 / a), jnp.zeros((b,), jnp.float32))
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return (x @ w + b).squeeze(-1)


def loss_fn(params, x, y):
    return jnp.mean(jnp.maximum(forward(params, x), 0) - forward(params, x) * y + jnp.log1p(jnp.exp(-jnp.abs(forward(params, x)))))


@jax.jit
def train_step(params, x, y, learning_rate):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads), loss


def telemetry():
    out = {"visible_devices": [str(d) for d in jax.devices()], "device_count": jax.device_count(),
           "platform": platform.platform(), "jax_version": jax.__version__}
    try:
        out["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        out["nvidia_smi"] = "not available"
    return out


def main():
    a = parse_args()
    X_train, X_val, X_test, y_train, y_val, y_test, n_rows, n_features = load_data(a.data, a.seed)
    device = jax.devices()[0]
    X_train, y_train = jax.device_put(X_train, device), jax.device_put(y_train, device)
    X_val, X_test = jax.device_put(X_val, device), jax.device_put(X_test, device)
    y_val, y_test = jax.device_put(y_val, device), jax.device_put(y_test, device)
    params = init_params(jax.random.PRNGKey(a.seed), n_features)
    batches = max(1, X_train.shape[0] // a.batch_size)

    # First call includes XLA compilation; later calls measure steady-state training only.
    compile_start = time.perf_counter()
    params, _ = train_step(params, X_train[:a.batch_size], y_train[:a.batch_size], a.learning_rate)
    jax.block_until_ready(params)
    compile_seconds = time.perf_counter() - compile_start

    step_times = []
    train_start = time.perf_counter()
    for epoch in range(a.epochs):
        order = np.random.default_rng(a.seed + epoch).permutation(X_train.shape[0])
        for i in range(batches):
            ids = order[i * a.batch_size:(i + 1) * a.batch_size]
            t0 = time.perf_counter()
            params, _ = train_step(params, X_train[ids], y_train[ids], a.learning_rate)
            jax.block_until_ready(params)
            step_times.append(time.perf_counter() - t0)
    training_seconds = time.perf_counter() - train_start

    val_prob = np.asarray(jax.nn.sigmoid(forward(params, X_val)))
    test_prob = np.asarray(jax.nn.sigmoid(forward(params, X_test)))
    val_acc = accuracy_score(np.asarray(y_val), val_prob >= 0.5)
    test_acc = accuracy_score(np.asarray(y_test), test_prob >= 0.5)
    payload = {"run_name": a.run_name, "dataset_rows": n_rows, "features": n_features, "epochs": a.epochs,
               "batch_size": a.batch_size, "steps": len(step_times), "compile_seconds": compile_seconds,
               "training_seconds": training_seconds, "mean_step_ms": float(np.mean(step_times) * 1000),
               "p95_step_ms": float(np.percentile(step_times, 95) * 1000), "steps_per_second": len(step_times) / training_seconds,
               "validation_accuracy": float(val_acc), "test_accuracy": float(test_acc),
               "test_roc_auc": float(roc_auc_score(np.asarray(y_test), test_prob)), "telemetry": telemetry()}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run_name}.json").write_text(json.dumps(payload, indent=2))
    flat = {k: v for k, v in payload.items() if k != "telemetry"}; flat["devices"] = "; ".join(payload["telemetry"]["visible_devices"])
    with (out / "benchmark_summary.csv").open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat.keys());
        if f.tell() == 0: w.writeheader()
        w.writerow(flat)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
