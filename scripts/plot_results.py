#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot experiment results for AWSeg.

This script is intended for the integrated main branch.

Expected result layout:
    outputs/results/
    ├── baseline/
    │   └── eval_val.json
    ├── loss/
    │   ├── ce/
    │   │   ├── eval_val.json
    │   │   ├── eval_val_fog.json
    │   │   ├── eval_val_night.json
    │   │   ├── eval_val_rain.json
    │   │   └── eval_val_snow.json
    │   └── ...
    ├── model/
    ├── augmentation/
    ├── enhancement/
    └── proposed/

Examples:
    python scripts/plot_results.py --group loss
    python scripts/plot_results.py --group augmentation
    python scripts/plot_results.py --group enhancement
    python scripts/plot_results.py --group proposed

    python scripts/plot_results.py \
        --group enhancement \
        --experiments gamma clahe gamma_clahe \
        --baseline-experiment none

Outputs:
    outputs/visualizations/<group>/
    ├── summary.csv
    ├── overall_miou.png
    ├── delta_miou.png
    ├── condition_miou.csv
    ├── condition_miou_heatmap.png
    ├── class_iou_matrix.csv
    ├── class_iou_heatmap.png
    └── class_iou_delta_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_ROOT = Path("outputs/results")
DEFAULT_OUTPUT_ROOT = Path("outputs/visualizations")

DEFAULT_CONDITIONS = ["fog", "night", "rain", "snow"]

DEFAULT_CLASS_ORDER = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

EXPERIMENT_ORDERS = {
    "loss": [
        "ce",
        "dice",
        "focal",
        "tversky",
        "focal_tversky",
        "ce_dice",
        "ce_tversky",
        "ce_lovasz",
        "ohem_tversky",
        "tversky_lovasz",
        "ce_focal_dice",
        "ce_focal_tversky",
    ],
    "model": [
        "unet",
        "segformer",
    ],
    "augmentation": [
        "none",
        "flip",
        "jitter",
        "class_crop",
        "weather_fog",
        "weather_rain",
        "weather_snow",
        "weather_night",
        "weather_mixed",
        "all",
    ],
    "enhancement": [
        "none",
        "gamma",
        "clahe",
        "gamma_clahe",
    ],
    "proposed": [
        "proposed",
    ],
}

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "ce": "CE",
    "dice": "Dice",
    "focal": "Focal",
    "tversky": "Tversky",
    "focal_tversky": "Focal Tversky",
    "ce_dice": "CE + Dice",
    "ce_tversky": "CE + Tversky",
    "ce_lovasz": "CE + Lovasz",
    "ohem_tversky": "OHEM + Tversky",
    "tversky_lovasz": "Tversky + Lovasz",
    "ce_focal_dice": "CE + Focal + Dice",
    "ce_focal_tversky": "CE + Focal Tversky",
    "unet": "U-Net",
    "segformer": "SegFormer",
    "none": "None",
    "flip": "Flip",
    "jitter": "Color jitter",
    "class_crop": "Class crop",
    "weather_fog": "Weather fog",
    "weather_rain": "Weather rain",
    "weather_snow": "Weather snow",
    "weather_night": "Weather night",
    "weather_mixed": "Weather mixed",
    "all": "All",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "gamma_clahe": "Gamma + CLAHE",
    "proposed": "Proposed",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read JSON: {path} ({exc})")
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def get_metric_record(data: dict[str, Any]) -> dict[str, Any]:
    """Return the part of a JSON that contains metrics.

    Supported examples:
        {"miou": 0.5, "class_iou": {...}}
        {"main": {"miou": 0.5, "class_iou": {...}}}
        {"overall": {"miou": 0.5}}
        {"metrics": {"miou": 0.5}}
    """
    for key in ("main", "overall", "metrics", "val"):
        value = data.get(key)
        if isinstance(value, dict) and any(
            metric_key in value for metric_key in ("miou", "mIoU", "mean_iou", "mean_IoU")
        ):
            return value
    return data


def get_miou(data: dict[str, Any] | None) -> float | None:
    if data is None:
        return None

    record = get_metric_record(data)
    for key in ("miou", "mIoU", "mean_iou", "mean_IoU", "best_miou", "val_miou"):
        value = as_float(record.get(key))
        if value is not None:
            return value

    # Some result files may keep scalar values under nested fields.
    for value in record.values():
        if isinstance(value, dict):
            nested = get_miou(value)
            if nested is not None:
                return nested

    return None


def get_num_samples(data: dict[str, Any] | None) -> int | None:
    if data is None:
        return None

    record = get_metric_record(data)
    for key in ("num_samples", "num_images", "n", "count"):
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def clean_class_name(name: Any) -> str:
    return str(name).strip()


def get_class_iou_from_json(data: dict[str, Any] | None) -> dict[str, float]:
    if data is None:
        return {}

    record = get_metric_record(data)

    candidates = [
        record.get("class_iou"),
        record.get("class_IoU"),
        record.get("class_ious"),
        record.get("per_class_iou"),
        data.get("class_iou"),
        data.get("class_IoU"),
        data.get("class_ious"),
        data.get("per_class_iou"),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            clean: dict[str, float] = {}
            for cls, value in candidate.items():
                numeric = as_float(value)
                if numeric is not None:
                    clean[clean_class_name(cls)] = numeric
            if clean:
                return clean

        if isinstance(candidate, list):
            clean = {}
            for index, value in enumerate(candidate):
                numeric = as_float(value)
                if numeric is not None:
                    cls = DEFAULT_CLASS_ORDER[index] if index < len(DEFAULT_CLASS_ORDER) else str(index)
                    clean[cls] = numeric
            if clean:
                return clean

    return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def get_class_iou_from_csv(exp_dir: Path) -> dict[str, float]:
    candidates = [
        exp_dir / "class_iou.csv",
        exp_dir / "class_ious.csv",
        exp_dir / "eval_val_class_iou.csv",
    ]

    for path in candidates:
        rows = read_csv_rows(path)
        if not rows:
            continue

        result: dict[str, float] = {}
        fieldnames = set(rows[0].keys())
        class_key = next(
            (key for key in ("class", "class_name", "name", "label") if key in fieldnames),
            None,
        )
        value_key = next(
            (key for key in ("iou", "IoU", "class_iou", "value", "score") if key in fieldnames),
            None,
        )

        if class_key is None or value_key is None:
            continue

        for row in rows:
            cls = clean_class_name(row.get(class_key, ""))
            value = as_float(row.get(value_key))
            if cls and value is not None:
                result[cls] = value

        if result:
            return result

    return {}


def get_condition_results_from_json(
    data: dict[str, Any] | None,
    conditions: list[str],
) -> dict[str, float]:
    if data is None:
        return {}

    record = get_metric_record(data)
    candidates = [
        record.get("condition_results"),
        record.get("conditions"),
        record.get("condition_iou"),
        record.get("condition_miou"),
        data.get("condition_results"),
        data.get("conditions"),
        data.get("condition_iou"),
        data.get("condition_miou"),
    ]

    result: dict[str, float] = {}

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        for condition, value in candidate.items():
            condition_key = str(condition).strip().lower()
            if condition_key not in conditions:
                continue

            if isinstance(value, dict):
                miou = get_miou(value)
            else:
                miou = as_float(value)

            if miou is not None:
                result[condition_key] = miou

    return result


def get_condition_results_from_csv(exp_dir: Path, conditions: list[str]) -> dict[str, float]:
    candidates = [
        exp_dir / "condition_iou.csv",
        exp_dir / "condition_miou.csv",
        exp_dir / "conditions.csv",
    ]

    for path in candidates:
        rows = read_csv_rows(path)
        if not rows:
            continue

        result: dict[str, float] = {}
        fieldnames = set(rows[0].keys())
        condition_key = next(
            (key for key in ("condition", "weather", "name") if key in fieldnames),
            None,
        )
        value_key = next(
            (key for key in ("miou", "mIoU", "mean_iou", "iou", "value", "score") if key in fieldnames),
            None,
        )

        if condition_key is None or value_key is None:
            continue

        for row in rows:
            condition = str(row.get(condition_key, "")).strip().lower()
            value = as_float(row.get(value_key))
            if condition in conditions and value is not None:
                result[condition] = value

        if result:
            return result

    return {}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").replace("-", " ").title())


def infer_experiments(group_dir: Path, group: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit

    if not group_dir.exists():
        raise FileNotFoundError(f"Result group directory not found: {group_dir}")

    found = [path.name for path in sorted(group_dir.iterdir()) if path.is_dir()]
    preferred = EXPERIMENT_ORDERS.get(group, [])

    ordered = [name for name in preferred if name in found]
    ordered.extend([name for name in found if name not in ordered])
    return ordered


def choose_baseline_experiment(
    group: str,
    experiments: list[str],
    requested: str | None,
) -> str | None:
    if requested:
        return requested if requested in experiments else None

    preferred_by_group = {
        "loss": "ce",
        "model": "unet",
        "augmentation": "none",
        "enhancement": "none",
        "proposed": "baseline",
        "baseline": "baseline",
    }

    preferred = preferred_by_group.get(group)
    if preferred in experiments:
        return preferred

    if experiments:
        return experiments[0]

    return None


def find_overall_json(exp_dir: Path) -> Path | None:
    candidates = [
        exp_dir / "eval_val.json",
        exp_dir / "eval.json",
        exp_dir / "metrics.json",
        exp_dir / "summary.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def find_condition_json(exp_dir: Path, condition: str) -> Path | None:
    candidates = [
        exp_dir / f"eval_val_{condition}.json",
        exp_dir / f"eval_{condition}.json",
        exp_dir / f"{condition}.json",
        exp_dir / f"metrics_{condition}.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def collect_experiment(
    group_dir: Path,
    experiment: str,
    conditions: list[str],
) -> dict[str, Any]:
    exp_dir = group_dir / experiment
    overall_json_path = find_overall_json(exp_dir)
    overall_data = maybe_load_json(overall_json_path) if overall_json_path else None

    condition_miou = get_condition_results_from_json(overall_data, conditions)
    csv_condition_miou = get_condition_results_from_csv(exp_dir, conditions)
    condition_miou.update(csv_condition_miou)

    for condition in conditions:
        if condition in condition_miou:
            continue

        condition_json_path = find_condition_json(exp_dir, condition)
        condition_data = maybe_load_json(condition_json_path) if condition_json_path else None
        condition_value = get_miou(condition_data)
        if condition_value is not None:
            condition_miou[condition] = condition_value

    class_iou = get_class_iou_from_json(overall_data)
    if not class_iou:
        class_iou = get_class_iou_from_csv(exp_dir)

    return {
        "experiment": experiment,
        "display_name": display_name(experiment),
        "dir": str(exp_dir),
        "overall_json": str(overall_json_path) if overall_json_path else "",
        "overall_miou": get_miou(overall_data),
        "num_samples": get_num_samples(overall_data),
        "condition_miou": condition_miou,
        "class_iou": class_iou,
    }


def collect_results(
    results_root: Path,
    group: str,
    experiments: list[str] | None,
    conditions: list[str],
) -> list[dict[str, Any]]:
    group_dir = results_root / group

    # Allow --group proposed even when outputs/results/proposed/eval_val.json exists
    # without a nested proposed/ directory.
    if group == "proposed" and (group_dir / "eval_val.json").exists():
        group_dir = results_root
        experiments = ["proposed"]

    experiment_names = infer_experiments(group_dir, group, experiments)

    rows = []
    for experiment in experiment_names:
        exp_dir = group_dir / experiment
        if not exp_dir.exists():
            print(f"[WARN] Skipping missing experiment directory: {exp_dir}")
            continue

        result = collect_experiment(group_dir, experiment, conditions)
        if result["overall_miou"] is None and not result["condition_miou"] and not result["class_iou"]:
            print(f"[WARN] No readable metrics found in: {exp_dir}")
            continue
        rows.append(result)

    if not rows:
        raise FileNotFoundError(
            f"No readable results found under {results_root / group}. "
            "Expected eval_val.json or related metric files."
        )

    return rows


def save_summary_csv(rows: list[dict[str, Any]], output_dir: Path, baseline: str | None) -> None:
    baseline_miou = None
    if baseline is not None:
        for row in rows:
            if row["experiment"] == baseline:
                baseline_miou = row["overall_miou"]
                break

    path = output_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "experiment",
            "display_name",
            "overall_miou",
            "delta_vs_baseline",
            "num_samples",
            "dir",
            "overall_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            overall_miou = row["overall_miou"]
            delta = None
            if baseline_miou is not None and overall_miou is not None:
                delta = overall_miou - baseline_miou

            writer.writerow(
                {
                    "experiment": row["experiment"],
                    "display_name": row["display_name"],
                    "overall_miou": "" if overall_miou is None else f"{overall_miou:.8f}",
                    "delta_vs_baseline": "" if delta is None else f"{delta:.8f}",
                    "num_samples": "" if row["num_samples"] is None else row["num_samples"],
                    "dir": row["dir"],
                    "overall_json": row["overall_json"],
                }
            )


def build_condition_matrix(rows: list[dict[str, Any]], conditions: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(conditions)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, condition in enumerate(conditions):
            value = row["condition_miou"].get(condition)
            if value is not None:
                matrix[i, j] = value
    return matrix


def save_condition_csv(
    rows: list[dict[str, Any]],
    conditions: list[str],
    output_dir: Path,
) -> None:
    path = output_dir / "condition_miou.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["experiment", "display_name", *conditions]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = {
                "experiment": row["experiment"],
                "display_name": row["display_name"],
            }
            for condition in conditions:
                value = row["condition_miou"].get(condition)
                out[condition] = "" if value is None else f"{value:.8f}"
            writer.writerow(out)


def build_class_order(rows: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for row in rows:
        for cls in row["class_iou"].keys():
            if cls not in found:
                found.append(cls)

    ordered = [cls for cls in DEFAULT_CLASS_ORDER if cls in found]
    ordered.extend([cls for cls in found if cls not in ordered])
    return ordered


def build_class_matrix(rows: list[dict[str, Any]], classes: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(classes)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, cls in enumerate(classes):
            value = row["class_iou"].get(cls)
            if value is not None:
                matrix[i, j] = value
    return matrix


def save_class_iou_csv(
    rows: list[dict[str, Any]],
    classes: list[str],
    output_dir: Path,
) -> None:
    if not classes:
        return

    path = output_dir / "class_iou_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["experiment", "display_name", *classes]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = {
                "experiment": row["experiment"],
                "display_name": row["display_name"],
            }
            for cls in classes:
                value = row["class_iou"].get(cls)
                out[cls] = "" if value is None else f"{value:.8f}"
            writer.writerow(out)


def add_bar_labels(ax: plt.Axes, values: list[float | None], fmt: str = "{:.4f}") -> None:
    for patch, value in zip(ax.patches, values):
        if value is None or math.isnan(value):
            continue
        height = patch.get_height()
        if math.isnan(height):
            continue
        offset = 0.006 if height >= 0 else -0.006
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + offset,
            fmt.format(value),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
            rotation=0,
        )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overall_miou(
    rows: list[dict[str, Any]],
    output_dir: Path,
    group: str,
    baseline: str | None,
    dpi: int,
) -> None:
    values = [row["overall_miou"] for row in rows]
    if all(value is None for value in values):
        print("[WARN] No overall mIoU found. Skipping overall_miou.png")
        return

    labels = [row["display_name"] for row in rows]
    x = np.arange(len(rows))
    y = [np.nan if value is None else value for value in values]

    width = max(8, min(22, 1.0 * len(rows) + 4))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(x, y)
    ax.set_title(f"{display_name(group)} experiment overall mIoU")
    ax.set_ylabel("mIoU")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, min(1.0, np.nanmax(y) + 0.08))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if baseline is not None:
        baseline_value = next(
            (row["overall_miou"] for row in rows if row["experiment"] == baseline),
            None,
        )
        if baseline_value is not None:
            ax.axhline(
                baseline_value,
                linestyle="--",
                linewidth=1.2,
                label=f"{display_name(baseline)}: {baseline_value:.4f}",
            )
            ax.legend(loc="best")

    add_bar_labels(ax, values)
    fig.tight_layout()
    save_figure(fig, output_dir / "overall_miou.png", dpi=dpi)


def plot_delta_miou(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline: str | None,
    dpi: int,
) -> None:
    if baseline is None:
        print("[WARN] No baseline experiment. Skipping delta_miou.png")
        return

    baseline_value = next(
        (row["overall_miou"] for row in rows if row["experiment"] == baseline),
        None,
    )
    if baseline_value is None:
        print("[WARN] Baseline mIoU is missing. Skipping delta_miou.png")
        return

    delta_rows = [
        row for row in rows if row["experiment"] != baseline and row["overall_miou"] is not None
    ]
    if not delta_rows:
        print("[WARN] No comparable experiments. Skipping delta_miou.png")
        return

    labels = [row["display_name"] for row in delta_rows]
    deltas = [float(row["overall_miou"]) - float(baseline_value) for row in delta_rows]
    x = np.arange(len(delta_rows))

    width = max(8, min(22, 1.0 * len(delta_rows) + 4))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(x, deltas)
    ax.axhline(0, linewidth=1.0)
    ax.set_title(f"ΔmIoU vs {display_name(baseline)}")
    ax.set_ylabel("ΔmIoU")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    add_bar_labels(ax, deltas, fmt="{:+.4f}")
    fig.tight_layout()
    save_figure(fig, output_dir / "delta_miou.png", dpi=dpi)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
    dpi: int,
    center_zero: bool = False,
    fmt: str = ".3f",
) -> None:
    if matrix.size == 0 or np.all(np.isnan(matrix)):
        print(f"[WARN] Empty matrix. Skipping {output_path.name}")
        return

    rows = max(4, 0.45 * len(row_labels) + 2)
    cols = max(8, 0.55 * len(col_labels) + 3)
    fig, ax = plt.subplots(figsize=(cols, rows))

    if center_zero:
        finite = matrix[np.isfinite(matrix)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel("ΔIoU" if center_zero else "IoU", rotation=270, labelpad=15)

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)


def plot_condition_miou(
    rows: list[dict[str, Any]],
    conditions: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    matrix = build_condition_matrix(rows, conditions)
    row_labels = [row["display_name"] for row in rows]
    plot_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=[condition.capitalize() for condition in conditions],
        title="Condition-wise mIoU",
        output_path=output_dir / "condition_miou_heatmap.png",
        dpi=dpi,
        center_zero=False,
        fmt=".3f",
    )


def plot_class_iou(
    rows: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
    baseline: str | None,
) -> None:
    classes = build_class_order(rows)
    save_class_iou_csv(rows, classes, output_dir)

    if not classes:
        print("[WARN] No class IoU found. Skipping class heatmaps.")
        return

    matrix = build_class_matrix(rows, classes)
    row_labels = [row["display_name"] for row in rows]

    plot_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=classes,
        title="Class-wise IoU",
        output_path=output_dir / "class_iou_heatmap.png",
        dpi=dpi,
        center_zero=False,
        fmt=".2f",
    )

    if baseline is None:
        return

    baseline_index = next(
        (index for index, row in enumerate(rows) if row["experiment"] == baseline),
        None,
    )
    if baseline_index is None or np.all(np.isnan(matrix[baseline_index])):
        return

    delta_matrix = matrix - matrix[baseline_index : baseline_index + 1, :]
    keep_indices = [index for index, row in enumerate(rows) if row["experiment"] != baseline]
    if not keep_indices:
        return

    plot_heatmap(
        matrix=delta_matrix[keep_indices],
        row_labels=[row_labels[index] for index in keep_indices],
        col_labels=classes,
        title=f"Class-wise ΔIoU vs {display_name(baseline)}",
        output_path=output_dir / "class_iou_delta_heatmap.png",
        dpi=dpi,
        center_zero=True,
        fmt="+.2f",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate result plots for baseline/loss/model/augmentation/enhancement/proposed experiments."
    )
    parser.add_argument(
        "--group",
        required=True,
        choices=["baseline", "loss", "model", "augmentation", "enhancement", "proposed"],
        help="Experiment group under outputs/results.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional experiment folder names. If omitted, all folders are scanned in a predefined order.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing result folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. Overrides --output-root/<group>.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Condition names to include in condition-wise plots.",
    )
    parser.add_argument(
        "--baseline-experiment",
        default=None,
        help="Experiment name used as baseline for delta plots. Auto-selected if omitted.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = ensure_dir(args.output_dir or (args.output_root / args.group))
    conditions = [condition.strip().lower() for condition in args.conditions]

    rows = collect_results(
        results_root=args.results_root,
        group=args.group,
        experiments=args.experiments,
        conditions=conditions,
    )

    baseline = choose_baseline_experiment(
        group=args.group,
        experiments=[row["experiment"] for row in rows],
        requested=args.baseline_experiment,
    )

    if baseline is None:
        print("[WARN] Baseline experiment could not be inferred.")

    save_summary_csv(rows, output_dir, baseline=baseline)
    save_condition_csv(rows, conditions, output_dir)

    plot_overall_miou(rows, output_dir, group=args.group, baseline=baseline, dpi=args.dpi)
    plot_delta_miou(rows, output_dir, baseline=baseline, dpi=args.dpi)
    plot_condition_miou(rows, conditions, output_dir, dpi=args.dpi)
    plot_class_iou(rows, output_dir, dpi=args.dpi, baseline=baseline)

    print(f"[OK] Plots and CSV summaries saved to: {output_dir}")


if __name__ == "__main__":
    main()
