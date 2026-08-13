#!/usr/bin/env python3
"""Build the final three-platform summary and presentation-ready SVG charts."""

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
RUNS = [
    ("Assigned CPU node", "cpu-ways-58-20260813-070716.json", "TFRT_CPU_0"),
    ("NVIDIA GH200", "churn-gpu-ways-58-20260813-074833.json", "cuda:0"),
    ("TPU v5e 2x4", "churn-tpu-ways-58-telemetry-final2-20260813-211541.json", "8 TPU devices"),
]
FIELDS = [
    "platform", "run_name", "dataset_rows", "features", "epochs", "batch_size",
    "steps", "compile_seconds", "data_loading_seconds", "device_transfer_seconds",
    "training_seconds", "mean_step_ms", "p95_step_ms", "steps_per_second",
    "measured_end_to_end_seconds", "validation_accuracy", "test_accuracy",
    "test_roc_auc", "device_count", "jax_version", "devices",
    "process_cpu_utilization_percent", "peak_host_rss_mib",
    "accelerator_mean_utilization_percent", "accelerator_peak_utilization_percent",
    "accelerator_peak_memory", "accelerator_memory_unit",
]


def load_runs():
    rows = []
    for platform, filename, device_label in RUNS:
        path = RESULTS / filename
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        telemetry = record["telemetry"]
        row = {field: record.get(field) for field in FIELDS}
        row.update(
            platform=platform,
            device_count=telemetry["device_count"],
            jax_version=telemetry["jax_version"],
            devices=device_label,
            process_cpu_utilization_percent=telemetry.get("process_cpu_utilization_percent"),
            peak_host_rss_mib=telemetry.get("peak_host_rss_mib"),
            accelerator_mean_utilization_percent=telemetry.get(
                "gpu_mean_utilization_percent",
                telemetry.get("tpu_mean_duty_cycle_percent")),
            accelerator_peak_utilization_percent=telemetry.get(
                "gpu_peak_utilization_percent",
                telemetry.get("tpu_peak_duty_cycle_percent")),
            accelerator_peak_memory=telemetry.get(
                "gpu_peak_memory_used_mib",
                telemetry.get("tpu_peak_hbm_used_gib")),
            accelerator_memory_unit=(
                "MiB VRAM" if "gpu_peak_memory_used_mib" in telemetry
                else "GiB HBM (tpu-info reported)" if "tpu_peak_hbm_used_gib" in telemetry
                else "MiB host RSS"),
        )
        if row["epochs"] != 200 or row["steps"] != 2600:
            raise ValueError(f"{filename} is not a comparable 200-epoch run")
        rows.append(row)
    return rows


def write_csv(rows):
    path = RESULTS / "benchmark_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(rows, field, title, unit, filename, decimals=2):
    values = [float(row[field]) for row in rows]
    labels = [row["platform"] for row in rows]
    colors = ["#2563EB", "#16A34A", "#9333EA"]
    width, height = 900, 520
    left, right, top, bottom = 110, 40, 80, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(values) * 1.16
    bar_w = 150
    gap = (plot_w - len(values) * bar_w) / (len(values) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#4B5563">{html.escape(unit)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_h - plot_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#6B7280">{value:.{decimals}f}</text>')
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap + index * (bar_w + gap)
        bar_h = plot_h * value / maximum
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-10:.1f}" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#111827">{value:.{decimals}f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">{html.escape(label)}</text>')
    parts.append('</svg>')
    (CHARTS / filename).write_text("\n".join(parts), encoding="utf-8")


def quality_chart(rows):
    labels = [row["platform"] for row in rows]
    accuracy = [100 * float(row["test_accuracy"]) for row in rows]
    auc = [100 * float(row["test_roc_auc"]) for row in rows]
    width, height = 900, 520
    left, right, top, bottom = 110, 40, 90, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    minimum, maximum = 50.0, 90.0
    group_w, bar_w = plot_w / 3, 82
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="450" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Predictive quality is similar, except GPU ROC-AUC</text>',
        '<rect x="315" y="55" width="14" height="14" fill="#0EA5E9"/><text x="337" y="67" font-family="Arial" font-size="13" fill="#374151">Accuracy</text>',
        '<rect x="465" y="55" width="14" height="14" fill="#F97316"/><text x="487" y="67" font-family="Arial" font-size="13" fill="#374151">ROC-AUC</text>',
    ]
    for tick in range(50, 91, 10):
        y = top + plot_h - plot_h * (tick - minimum) / (maximum - minimum)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#6B7280">{tick}%</text>')
    for index, label in enumerate(labels):
        center = left + group_w * (index + 0.5)
        for offset, value, color in [(-bar_w/2, accuracy[index], "#0EA5E9"), (bar_w/2, auc[index], "#F97316")]:
            x = center + offset - bar_w/2
            bar_h = plot_h * (value - minimum) / (maximum - minimum)
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#111827">{value:.1f}%</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+30}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">{html.escape(label)}</text>')
    parts.append('</svg>')
    (CHARTS / "predictive_quality.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    write_csv(rows)
    bar_chart(rows, "training_seconds", "CPU wins on training time", "seconds (lower is better)", "training_time.svg")
    bar_chart(rows, "steps_per_second", "CPU delivers the highest throughput", "training steps/second (higher is better)", "throughput.svg")
    quality_chart(rows)
    print(f"Wrote {RESULTS / 'benchmark_summary.csv'} and 3 charts in {CHARTS}")


if __name__ == "__main__":
    main()
