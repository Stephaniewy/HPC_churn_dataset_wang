"""Reproducible JAX/XLA DNN churn benchmark for CPU, GH200 GPU, and TPU v5e."""
import argparse
import csv
import io
import json
import os
import platform
import re
import resource
import subprocess
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


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
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        feature_names = [name for name in reader.fieldnames if name != "churned"]
        rows, labels = [], []
        for row in reader:
            rows.append([float(row[name]) if row[name].strip() else np.nan for name in feature_names])
            labels.append(float(row["churned"].strip().upper() == "TRUE"))
    X, y = np.asarray(rows, np.float32), np.asarray(labels, np.float32)
    medians = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), medians, X).astype(np.float32)

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_test = round(len(indices) * 0.15)
        n_val = round(len(indices) * 0.15)
        splits["test"].extend(indices[:n_test])
        splits["val"].extend(indices[n_test:n_test + n_val])
        splits["train"].extend(indices[n_test + n_val:])
    for indices in splits.values():
        rng.shuffle(indices)
    train_idx = np.asarray(splits["train"])
    val_idx = np.asarray(splits["val"])
    test_idx = np.asarray(splits["test"])
    mean, std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    std = np.where(std == 0, 1.0, std)
    scaled = ((X - mean) / std).astype(np.float32)
    return (scaled[train_idx], scaled[val_idx], scaled[test_idx],
            y[train_idx], y[val_idx], y[test_idx], len(y), X.shape[1])


def binary_roc_auc(y_true, scores):
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    if not len(positive) or not len(negative):
        raise ValueError("ROC-AUC requires both classes")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


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
    logits = forward(params, x)
    return jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))


@jax.jit
def train_step(params, x, y, learning_rate):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads), loss


def sample_nvidia(stop_event, samples):
    query = "name,utilization.gpu,memory.used,memory.total"
    while not stop_event.is_set():
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL).strip()
            for line in raw.splitlines():
                name, utilization, used, total = (part.strip() for part in line.split(","))
                samples.append({"timestamp_seconds": time.time(), "name": name,
                                "utilization_percent": float(utilization),
                                "memory_used_mib": float(used), "memory_total_mib": float(total)})
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return
        stop_event.wait(0.5)


def parse_tpu_info(raw):
    """Parse selected tpu-info tables into one structured sample."""
    sample = {"timestamp_seconds": time.time()}
    duty, tensorcore, hbm_used, hbm_total = [], [], [], []
    section = None
    for line in raw.splitlines():
        # Rich emits Unicode borders and can emit ANSI styling. Normalize both
        # forms before parsing table cells.
        stripped = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
        stripped = stripped.replace("│", "|")
        if stripped == "TPU Duty Cycle":
            section = "duty"
            continue
        if stripped == "TPU HBM Usage":
            section = "hbm"
            continue
        if stripped == "TensorCore Utilization":
            section = "tensorcore"
            continue
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or not cells[0].isdigit():
            continue
        try:
            if section == "duty" and len(cells) >= 2:
                duty.append(float(cells[1].rstrip("%")))
            elif section == "tensorcore" and len(cells) >= 2:
                tensorcore.append(float(cells[1].rstrip("%")))
            elif section == "hbm" and len(cells) >= 2:
                match = re.search(r"([0-9.]+)\s*GiB\s*/\s*([0-9.]+)\s*GiB", cells[1])
                if match:
                    hbm_used.append(float(match.group(1)))
                    hbm_total.append(float(match.group(2)))
        except ValueError:
            continue
    if duty:
        sample["duty_cycle_percent"] = duty
    if tensorcore:
        sample["tensorcore_utilization_percent"] = tensorcore
    if hbm_used:
        sample["hbm_used_gib"] = hbm_used
        sample["hbm_total_gib"] = hbm_total
    return sample


def sample_tpu(stop_event, samples, diagnostics):
    """Poll libtpu runtime metrics while the JAX workload is active."""
    # Duty cycle and HBM are required here. Some older libtpu runtimes reject
    # the whole request when the optional TensorCore metric is also included.
    commands = [
        ["tpu-info", "--metric", "duty_cycle_percent"],
        ["tpu-info", "--metric", "hbm_usage"],
    ]
    while not stop_event.is_set():
        try:
            raw = "\n".join(
                subprocess.check_output(
                    command, text=True, stderr=subprocess.STDOUT, timeout=5)
                for command in commands)
            diagnostics["commands"] = commands
            diagnostics["last_output"] = raw[-12000:]
            sample = parse_tpu_info(raw)
            if len(sample) > 1:
                samples.append(sample)
        except subprocess.CalledProcessError as error:
            diagnostics["commands"] = commands
            diagnostics["error"] = f"exit status {error.returncode}"
            diagnostics["last_output"] = (error.output or "")[-12000:]
            return
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as error:
            diagnostics["commands"] = commands
            diagnostics["error"] = repr(error)
            return
        stop_event.wait(1.0)


def telemetry(gpu_samples, tpu_samples, tpu_diagnostics, cpu_seconds, measured_wall_seconds):
    affinity_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out = {"visible_devices": [str(d) for d in jax.devices()], "device_count": jax.device_count(),
           "platform": platform.platform(), "jax_version": jax.__version__,
           "cpu_affinity_count": affinity_count, "process_cpu_seconds": cpu_seconds,
           "process_cpu_utilization_percent": 100.0 * cpu_seconds / measured_wall_seconds / affinity_count,
           "peak_host_rss_mib": usage.ru_maxrss / 1024.0,
           "gpu_sample_interval_seconds": 0.5, "gpu_samples": gpu_samples,
           "tpu_sample_interval_seconds": 1.0, "tpu_samples": tpu_samples,
           "tpu_info_diagnostics": tpu_diagnostics}
    if gpu_samples:
        out["gpu_mean_utilization_percent"] = float(np.mean([s["utilization_percent"] for s in gpu_samples]))
        out["gpu_peak_utilization_percent"] = float(max(s["utilization_percent"] for s in gpu_samples))
        out["gpu_peak_memory_used_mib"] = float(max(s["memory_used_mib"] for s in gpu_samples))
    else:
        out["gpu_sampling"] = "nvidia-smi not available"
    if tpu_samples:
        duty = [value for sample in tpu_samples for value in sample.get("duty_cycle_percent", [])]
        tensorcore = [value for sample in tpu_samples for value in sample.get("tensorcore_utilization_percent", [])]
        hbm_used = [value for sample in tpu_samples for value in sample.get("hbm_used_gib", [])]
        hbm_total = [value for sample in tpu_samples for value in sample.get("hbm_total_gib", [])]
        if duty:
            out["tpu_mean_duty_cycle_percent"] = float(np.mean(duty))
            out["tpu_peak_duty_cycle_percent"] = float(max(duty))
        if tensorcore:
            out["tpu_mean_tensorcore_utilization_percent"] = float(np.mean(tensorcore))
            out["tpu_peak_tensorcore_utilization_percent"] = float(max(tensorcore))
        if hbm_used:
            out["tpu_peak_hbm_used_gib"] = float(max(hbm_used))
            out["tpu_hbm_total_gib_per_chip"] = float(max(hbm_total))
    elif any(getattr(device, "platform", "") == "tpu" for device in jax.devices()):
        out["tpu_sampling"] = "tpu-info returned no supported runtime metrics"
    return out


def append_summary(path, row):
    """Append a row while safely extending an older summary schema."""
    fields = list(row)
    prior = []
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            prior = list(reader)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prior)
        writer.writerow(row)


def main():
    a = parse_args()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    data_start = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test, n_rows, n_features = load_data(a.data, a.seed)
    data_loading_seconds = time.perf_counter() - data_start
    device = jax.devices()[0]
    transfer_start = time.perf_counter()
    X_train, y_train = jax.device_put(X_train, device), jax.device_put(y_train, device)
    X_val, X_test = jax.device_put(X_val, device), jax.device_put(X_test, device)
    y_val, y_test = jax.device_put(y_val, device), jax.device_put(y_test, device)
    jax.block_until_ready((X_train, X_val, X_test, y_train, y_val, y_test))
    device_transfer_seconds = time.perf_counter() - transfer_start
    params = init_params(jax.random.PRNGKey(a.seed), n_features)
    batches = max(1, X_train.shape[0] // a.batch_size)

    gpu_samples = []
    tpu_samples = []
    tpu_diagnostics = {}
    stop_sampling = threading.Event()
    sampler = threading.Thread(target=sample_nvidia, args=(stop_sampling, gpu_samples), daemon=True)
    sampler.start()
    tpu_sampler = None
    if getattr(device, "platform", "") == "tpu":
        tpu_sampler = threading.Thread(
            target=sample_tpu,
            args=(stop_sampling, tpu_samples, tpu_diagnostics), daemon=True)
        tpu_sampler.start()

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
    val_y, test_y = np.asarray(y_val), np.asarray(y_test)
    val_acc = np.mean(val_y == (val_prob >= 0.5))
    test_acc = np.mean(test_y == (test_prob >= 0.5))
    measured_wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    # LibTPU metrics refresh over a multi-second interval. Give a TPU-only
    # sampler a short post-measurement window to return its first real sample;
    # this wait is intentionally excluded from benchmark end-to-end time.
    if tpu_sampler:
        sample_deadline = time.monotonic() + 6.0
        while not tpu_samples and time.monotonic() < sample_deadline:
            time.sleep(0.1)
    stop_sampling.set()
    sampler.join(timeout=2)
    if tpu_sampler:
        tpu_sampler.join(timeout=6)
    payload = {"run_name": a.run_name, "dataset_rows": n_rows, "features": n_features, "epochs": a.epochs,
               "batch_size": a.batch_size, "steps": len(step_times), "compile_seconds": compile_seconds,
               "data_loading_seconds": data_loading_seconds, "device_transfer_seconds": device_transfer_seconds,
               "training_seconds": training_seconds, "mean_step_ms": float(np.mean(step_times) * 1000),
               "p95_step_ms": float(np.percentile(step_times, 95) * 1000), "steps_per_second": len(step_times) / training_seconds,
               "measured_end_to_end_seconds": measured_wall_seconds,
               "validation_accuracy": float(val_acc), "test_accuracy": float(test_acc),
               "test_roc_auc": binary_roc_auc(test_y, test_prob),
               "telemetry": telemetry(gpu_samples, tpu_samples, tpu_diagnostics,
                                      cpu_seconds, measured_wall_seconds)}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run_name}.json").write_text(json.dumps(payload, indent=2))
    flat = {k: v for k, v in payload.items() if k != "telemetry"}; flat["devices"] = "; ".join(payload["telemetry"]["visible_devices"])
    append_summary(out / "benchmark_summary.csv", flat)
    print("-----BEGIN BENCHMARK JSON-----")
    print(json.dumps(payload, indent=2))
    print("-----END BENCHMARK JSON-----")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=flat.keys())
    writer.writeheader(); writer.writerow(flat)
    print("-----BEGIN BENCHMARK CSV-----")
    print(stream.getvalue().strip())
    print("-----END BENCHMARK CSV-----")


if __name__ == "__main__":
    main()
