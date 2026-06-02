#!/usr/bin/env python
"""Detailed error analysis for ACDC semantic segmentation.

This script analyzes either the whole validation split or a selected weather
condition in sangchun1/Adverse-Weather-Segmentation.

It produces:
1) class-wise IoU comparison: overall val vs. condition val
   - skipped when --condition none is used
2) confusion matrix and top confusion pairs
3) per-image error summary
4) boundary error analysis using GT semantic-boundary bands
5) dynamic-object -> background confusion analysis
6) representative visualization examples with error maps

Examples:
    # Whole validation-set baseline analysis without condition-vs-overall IoU drop
    python scripts/analyze_errors.py \
      --config configs/baseline.yaml \
      --checkpoint outputs/checkpoints/baseline/best_miou.pth \
      --condition none \
      --device cuda:1

    # Night-specific analysis with overall-val vs. night-val IoU drop
    python scripts/analyze_errors.py \
      --config configs/night.yaml \
      --checkpoint outputs/checkpoints/night/best_miou.pth \
      --condition night \
      --overall-result outputs/results/baseline/eval_val.json \
      --condition-result outputs/results/night/eval_val.json \
      --output-dir outputs/analysis/night
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Allow `python scripts/analyze_errors.py` without relying only on pip -e .
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from awseg.dataset import build_dataset, get_class_names  # noqa: E402
from awseg.models import build_model  # noqa: E402
from awseg.utils import get_device, load_config  # noqa: E402
from awseg.visualize import (  # noqa: E402
    colorize_mask,
    denormalize_image,
    make_overlay,
    safe_filename,
)


DYNAMIC_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

BACKGROUND_CLASSES = [
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
]

# Error-map colors used only for diagnostic visualizations.
ERROR_COLORS = {
    "correct": np.array([0, 0, 0], dtype=np.uint8),
    "other_error": np.array([0, 128, 255], dtype=np.uint8),
    "boundary_error": np.array([255, 0, 0], dtype=np.uint8),
    "dynamic_to_background": np.array([255, 255, 0], dtype=np.uint8),
}

ALL_CONDITION_VALUES = {"", "none", "all", "overall", "val", "baseline"}


def is_all_condition(condition: str | None) -> bool:
    """Return True when all samples in the split should be analyzed."""
    if condition is None:
        return True
    return str(condition).strip().lower() in ALL_CONDITION_VALUES


def get_analysis_name(condition: str | None) -> str:
    """Use 'baseline' for whole-val analysis; otherwise use the condition name."""
    if is_all_condition(condition):
        return "baseline"
    name = str(condition).strip().lower()
    return re.sub(r"[^a-zA-Z0-9_.=-]+", "_", name) or "condition"


def get_analysis_label(condition: str | None) -> str:
    if is_all_condition(condition):
        return "whole val"
    return str(condition).strip()


@dataclass
class ImageErrorRecord:
    dataset_index: int
    image_path: str
    label_path: str
    condition: str
    image_miou: float
    pixel_error_rate: float
    boundary_error_rate: float
    interior_error_rate: float
    dynamic_to_background_error_rate: float
    valid_pixels: int
    boundary_pixels: int
    interior_pixels: int
    gt_dynamic_pixels: int
    error_pixels: int
    boundary_error_pixels: int
    interior_error_pixels: int
    dynamic_to_background_error_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze segmentation errors for a selected condition, "
            "or for the whole split when --condition none is used."
        )
    )
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Model checkpoint for generating predictions. "
            "If omitted, checkpoint.save_dir/save_best_name from the config is used. "
            "For baseline analysis, pass outputs/checkpoints/baseline/best_miou.pth "
            "or make sure configs/baseline.yaml points to that checkpoint directory."
        ),
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--condition",
        type=str,
        default="none",
        help=(
            "Condition to analyze, e.g. night/fog/rain/snow. "
            "Use 'none' to analyze the whole split without condition filtering."
        ),
    )
    parser.add_argument(
        "--overall-result",
        type=str,
        default="outputs/results/baseline/eval_val.json",
        help="Evaluation JSON for the whole validation set.",
    )
    parser.add_argument(
        "--condition-result",
        "--night-result",
        dest="condition_result",
        type=str,
        default=None,
        help=(
            "Evaluation JSON for condition-specific validation result. "
            "If omitted, outputs/results/<condition>/eval_val.json is used. "
            "Ignored when --condition none is used. "
            "--night-result is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: outputs/analysis/<condition>; "
            "outputs/analysis/baseline when --condition none."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of representative images to save for each category.",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=2,
        help="Dilation radius in pixels around GT semantic boundaries.",
    )
    parser.add_argument(
        "--min-dynamic-pixels",
        type=int,
        default=50,
        help=(
            "Minimum GT dynamic-object pixels required for selecting "
            "dynamic-to-background example images."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit for selected samples. Default: analyze all selected samples.",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda:1 or cpu. Default: awseg.utils.get_device().",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sanitize_class_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def class_name_to_id(class_names: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, name in enumerate(class_names):
        mapping[name] = idx
        mapping[sanitize_class_name(name)] = idx
    return mapping


def ids_from_names(names: Iterable[str], class_names: list[str]) -> list[int]:
    mapping = class_name_to_id(class_names)
    ids: list[int] = []
    missing: list[str] = []
    for name in names:
        key = sanitize_class_name(name)
        if key in mapping:
            ids.append(mapping[key])
        elif name in mapping:
            ids.append(mapping[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Unknown class names: {missing}. Available classes: {class_names}")
    return ids


def normalize_class_iou_dict(result_json: dict[str, Any], class_names: list[str]) -> dict[str, float]:
    class_iou = result_json.get("class_iou")
    if class_iou is None:
        raise KeyError("Result JSON does not contain `class_iou`.")

    if isinstance(class_iou, dict):
        normalized = {}
        for class_name in class_names:
            candidates = [class_name, sanitize_class_name(class_name)]
            value = None
            for key in candidates:
                if key in class_iou:
                    value = class_iou[key]
                    break
            normalized[class_name] = float("nan") if value is None else float(value)
        return normalized

    if isinstance(class_iou, list):
        if len(class_iou) != len(class_names):
            raise ValueError(
                f"class_iou length mismatch: {len(class_iou)} vs {len(class_names)} classes"
            )
        return {
            class_name: float(value) if value is not None else float("nan")
            for class_name, value in zip(class_names, class_iou)
        }

    raise TypeError(f"Unsupported class_iou type: {type(class_iou)}")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_class_iou_drop_outputs(
    overall_result_path: str | Path,
    condition_result_path: str | Path,
    class_names: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    overall_result = load_json(overall_result_path)
    condition_result = load_json(condition_result_path)

    overall_iou = normalize_class_iou_dict(overall_result, class_names)
    condition_iou = normalize_class_iou_dict(condition_result, class_names)

    rows: list[dict[str, Any]] = []
    for class_name in class_names:
        overall_value = overall_iou[class_name]
        condition_value = condition_iou[class_name]
        if math.isnan(overall_value) or math.isnan(condition_value):
            drop = float("nan")
            condition_minus_overall = float("nan")
        else:
            drop = overall_value - condition_value
            condition_minus_overall = condition_value - overall_value
        rows.append(
            {
                "class_id": class_names.index(class_name),
                "class_name": class_name,
                "overall_val_iou": overall_value,
                "condition_iou": condition_value,
                "iou_drop_overall_minus_condition": drop,
                "condition_minus_overall": condition_minus_overall,
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -1e9
            if math.isnan(float(r["iou_drop_overall_minus_condition"]))
            else float(r["iou_drop_overall_minus_condition"])
        ),
        reverse=True,
    )
    write_csv(
        output_dir / "class_iou_drop.csv",
        rows_sorted,
        fieldnames=[
            "class_id",
            "class_name",
            "overall_val_iou",
            "condition_iou",
            "iou_drop_overall_minus_condition",
            "condition_minus_overall",
        ],
    )

    plot_dir = ensure_dir(output_dir / "plots")
    plot_class_iou_comparison(rows, plot_dir / "class_iou_overall_vs_condition.png")
    plot_class_iou_drop(rows_sorted, plot_dir / "class_iou_drop_overall_minus_condition.png")

    return rows_sorted


def plot_class_iou_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    class_names = [str(r["class_name"]) for r in rows]
    overall = np.array([float(r["overall_val_iou"]) for r in rows], dtype=np.float32)
    condition = np.array([float(r["condition_iou"]) for r in rows], dtype=np.float32)

    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.55), 5))
    ax.bar(x - width / 2, overall, width, label="overall val")
    ax.bar(x + width / 2, condition, width, label="condition val")
    ax.set_ylabel("IoU")
    ax.set_title("Class-wise IoU: overall val vs. condition")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_iou_drop(rows_sorted: list[dict[str, Any]], output_path: Path) -> None:
    valid_rows = [
        r for r in rows_sorted if not math.isnan(float(r["iou_drop_overall_minus_condition"]))
    ]
    names = [str(r["class_name"]) for r in valid_rows]
    drops = np.array([float(r["iou_drop_overall_minus_condition"]) for r in valid_rows])

    fig_height = max(5, 0.35 * len(valid_rows))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(valid_rows))
    ax.barh(y, drops)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("IoU drop = overall val IoU - condition IoU")
    ax.set_title("Classes most degraded in selected condition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_checkpoint_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    checkpoint_config = config.get("checkpoint", {})
    save_dir = checkpoint_config.get("save_dir")
    save_best_name = checkpoint_config.get("save_best_name", "best_miou.pth")
    if not save_dir:
        raise ValueError(
            "--checkpoint was not given and config.checkpoint.save_dir is unavailable."
        )
    return Path(save_dir) / str(save_best_name)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


def make_semantic_boundary_band(
    gt: np.ndarray,
    valid_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Return a dilated band around all GT semantic boundaries."""
    boundary = np.zeros_like(valid_mask, dtype=bool)

    vertical_diff = (
        valid_mask[:-1, :]
        & valid_mask[1:, :]
        & (gt[:-1, :] != gt[1:, :])
    )
    boundary[:-1, :] |= vertical_diff
    boundary[1:, :] |= vertical_diff

    horizontal_diff = (
        valid_mask[:, :-1]
        & valid_mask[:, 1:]
        & (gt[:, :-1] != gt[:, 1:])
    )
    boundary[:, :-1] |= horizontal_diff
    boundary[:, 1:] |= horizontal_diff

    if radius <= 0:
        return boundary & valid_mask

    kernel_size = 2 * radius + 1
    boundary_tensor = torch.from_numpy(boundary.astype(np.float32))[None, None]
    dilated = F.max_pool2d(
        boundary_tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )[0, 0]
    return (dilated.numpy() > 0) & valid_mask


def compute_confusion_matrix_np(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    valid = gt != ignore_index
    valid &= gt >= 0
    valid &= gt < num_classes
    valid &= pred >= 0
    valid &= pred < num_classes
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    encoded = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    conf = np.bincount(encoded, minlength=num_classes * num_classes)
    return conf.reshape(num_classes, num_classes).astype(np.int64)


def compute_iou_from_confusion(conf: np.ndarray) -> tuple[float, np.ndarray]:
    conf_f = conf.astype(np.float64)
    tp = np.diag(conf_f)
    fp = conf_f.sum(axis=0) - tp
    fn = conf_f.sum(axis=1) - tp
    union = tp + fp + fn
    class_iou = np.full(conf.shape[0], np.nan, dtype=np.float64)
    valid = union > 0
    class_iou[valid] = tp[valid] / union[valid]
    miou = float(np.nanmean(class_iou)) if np.any(valid) else 0.0
    return miou, class_iou


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else float("nan")


def format_rate(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def analyze_predictions(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    condition: str,
    output_dir: Path,
    boundary_radius: int,
    max_samples: int | None,
    device: torch.device,
) -> tuple[list[ImageErrorRecord], np.ndarray, dict[str, np.ndarray]]:
    dataset = build_dataset(config, split=split)
    class_names = get_class_names()
    num_classes = int(config["data"].get("num_classes", len(class_names)))
    ignore_index = int(config["data"].get("ignore_index", 255))

    if is_all_condition(condition):
        selected_indices = list(range(len(dataset)))
    else:
        selected_indices = [
            idx
            for idx, sample in enumerate(dataset.samples)
            if str(sample.get("condition", "unknown")) == condition
        ]

    if max_samples is not None:
        selected_indices = selected_indices[:max_samples]
    if len(selected_indices) == 0:
        available = sorted({str(s.get("condition", "unknown")) for s in dataset.samples})
        raise ValueError(
            f"No samples found for condition={condition!r}. Available conditions: {available}"
        )

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Selected samples ({get_analysis_label(condition)}): "
        f"{len(selected_indices)} / {len(dataset)}"
    )

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")
    model.eval()

    dynamic_ids = ids_from_names(DYNAMIC_CLASSES, class_names)
    background_ids = ids_from_names(BACKGROUND_CLASSES, class_names)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    boundary_pixels_by_class = np.zeros(num_classes, dtype=np.int64)
    boundary_errors_by_class = np.zeros(num_classes, dtype=np.int64)
    records: list[ImageErrorRecord] = []

    with torch.no_grad():
        for local_idx, dataset_index in enumerate(selected_indices, start=1):
            sample = dataset[dataset_index]
            if "mask" not in sample:
                raise ValueError(
                    f"Sample does not contain mask. split={split}, index={dataset_index}"
                )
            image_tensor = sample["image"]
            gt_tensor = sample["mask"]
            logits = model(image_tensor.unsqueeze(0).to(device, non_blocking=True))
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)
            gt = gt_tensor.detach().cpu().numpy().astype(np.int64)

            valid = gt != ignore_index
            valid &= gt >= 0
            valid &= gt < num_classes
            valid &= pred >= 0
            valid &= pred < num_classes
            wrong = (pred != gt) & valid

            conf = compute_confusion_matrix_np(pred, gt, num_classes, ignore_index)
            confusion += conf
            image_miou, _ = compute_iou_from_confusion(conf)

            boundary_band = make_semantic_boundary_band(
                gt=gt,
                valid_mask=valid,
                radius=boundary_radius,
            )
            interior = valid & ~boundary_band

            dynamic_gt = np.isin(gt, dynamic_ids) & valid
            pred_background = np.isin(pred, background_ids) & valid
            dynamic_to_background = dynamic_gt & pred_background

            for class_id in range(num_classes):
                class_boundary = boundary_band & (gt == class_id)
                boundary_pixels_by_class[class_id] += int(class_boundary.sum())
                boundary_errors_by_class[class_id] += int((wrong & class_boundary).sum())

            valid_pixels = int(valid.sum())
            boundary_pixels = int(boundary_band.sum())
            interior_pixels = int(interior.sum())
            gt_dynamic_pixels = int(dynamic_gt.sum())
            error_pixels = int(wrong.sum())
            boundary_error_pixels = int((wrong & boundary_band).sum())
            interior_error_pixels = int((wrong & interior).sum())
            dynamic_to_background_error_pixels = int(dynamic_to_background.sum())

            records.append(
                ImageErrorRecord(
                    dataset_index=dataset_index,
                    image_path=str(sample.get("image_path", "")),
                    label_path=str(sample.get("label_path", "")),
                    condition=str(sample.get("condition", condition)),
                    image_miou=image_miou,
                    pixel_error_rate=safe_ratio(error_pixels, valid_pixels),
                    boundary_error_rate=safe_ratio(boundary_error_pixels, boundary_pixels),
                    interior_error_rate=safe_ratio(interior_error_pixels, interior_pixels),
                    dynamic_to_background_error_rate=safe_ratio(
                        dynamic_to_background_error_pixels,
                        gt_dynamic_pixels,
                    ),
                    valid_pixels=valid_pixels,
                    boundary_pixels=boundary_pixels,
                    interior_pixels=interior_pixels,
                    gt_dynamic_pixels=gt_dynamic_pixels,
                    error_pixels=error_pixels,
                    boundary_error_pixels=boundary_error_pixels,
                    interior_error_pixels=interior_error_pixels,
                    dynamic_to_background_error_pixels=dynamic_to_background_error_pixels,
                )
            )

            if local_idx == 1 or local_idx % 50 == 0 or local_idx == len(selected_indices):
                print(f"Analyzed {local_idx:4d}/{len(selected_indices)} samples")

    extras = {
        "boundary_pixels_by_class": boundary_pixels_by_class,
        "boundary_errors_by_class": boundary_errors_by_class,
    }
    return records, confusion, extras


def records_to_rows(records: list[ImageErrorRecord]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_index": r.dataset_index,
            "condition": r.condition,
            "image_path": r.image_path,
            "label_path": r.label_path,
            "image_miou": r.image_miou,
            "pixel_error_rate": r.pixel_error_rate,
            "boundary_error_rate": r.boundary_error_rate,
            "interior_error_rate": r.interior_error_rate,
            "dynamic_to_background_error_rate": r.dynamic_to_background_error_rate,
            "valid_pixels": r.valid_pixels,
            "boundary_pixels": r.boundary_pixels,
            "interior_pixels": r.interior_pixels,
            "gt_dynamic_pixels": r.gt_dynamic_pixels,
            "error_pixels": r.error_pixels,
            "boundary_error_pixels": r.boundary_error_pixels,
            "interior_error_pixels": r.interior_error_pixels,
            "dynamic_to_background_error_pixels": r.dynamic_to_background_error_pixels,
        }
        for r in records
    ]


def save_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    analysis_name: str,
) -> None:
    rows = []
    for gt_id, gt_name in enumerate(class_names):
        row = {"gt_class_id": gt_id, "gt_class_name": gt_name}
        for pred_id, pred_name in enumerate(class_names):
            row[f"pred_{pred_id}_{sanitize_class_name(pred_name)}"] = int(
                confusion[gt_id, pred_id]
            )
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / f"confusion_matrix_{analysis_name}.csv", rows, fieldnames)


def make_class_error_summary(
    confusion: np.ndarray,
    class_names: list[str],
    boundary_pixels_by_class: np.ndarray,
    boundary_errors_by_class: np.ndarray,
    output_dir: Path,
    analysis_name: str,
) -> list[dict[str, Any]]:
    dynamic_ids = set(ids_from_names(DYNAMIC_CLASSES, class_names))
    background_ids = ids_from_names(BACKGROUND_CLASSES, class_names)

    conf_f = confusion.astype(np.float64)
    tp = np.diag(conf_f)
    gt_pixels = conf_f.sum(axis=1)
    pred_pixels = conf_f.sum(axis=0)
    fp = pred_pixels - tp
    fn = gt_pixels - tp
    union = tp + fp + fn

    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        iou = safe_ratio(tp[class_id], union[class_id])
        recall = safe_ratio(tp[class_id], gt_pixels[class_id])
        precision = safe_ratio(tp[class_id], pred_pixels[class_id])
        class_error_rate = safe_ratio(fn[class_id], gt_pixels[class_id])
        boundary_error_rate = safe_ratio(
            int(boundary_errors_by_class[class_id]),
            int(boundary_pixels_by_class[class_id]),
        )
        dynamic_to_background_rate = float("nan")
        if class_id in dynamic_ids:
            dynamic_to_background_rate = safe_ratio(
                int(confusion[class_id, background_ids].sum()),
                int(gt_pixels[class_id]),
            )

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": int(gt_pixels[class_id]),
                "pred_pixels": int(pred_pixels[class_id]),
                "true_positive": int(tp[class_id]),
                "false_positive": int(fp[class_id]),
                "false_negative": int(fn[class_id]),
                "iou": iou,
                "recall": recall,
                "precision": precision,
                "class_error_rate_gt_missed": class_error_rate,
                "boundary_pixels": int(boundary_pixels_by_class[class_id]),
                "boundary_error_pixels": int(boundary_errors_by_class[class_id]),
                "boundary_error_rate": boundary_error_rate,
                "dynamic_to_background_rate": dynamic_to_background_rate,
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -1.0 if math.isnan(float(r["class_error_rate_gt_missed"])) else float(r["class_error_rate_gt_missed"])
        ),
        reverse=True,
    )
    write_csv(
        output_dir / f"class_error_summary_{analysis_name}.csv",
        rows_sorted,
        fieldnames=[
            "class_id",
            "class_name",
            "gt_pixels",
            "pred_pixels",
            "true_positive",
            "false_positive",
            "false_negative",
            "iou",
            "recall",
            "precision",
            "class_error_rate_gt_missed",
            "boundary_pixels",
            "boundary_error_pixels",
            "boundary_error_rate",
            "dynamic_to_background_rate",
        ],
    )
    return rows_sorted


def make_top_confusion_pairs(
    confusion: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    analysis_name: str,
    top_n: int = 30,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt_pixels = confusion.sum(axis=1)
    for gt_id, gt_name in enumerate(class_names):
        for pred_id, pred_name in enumerate(class_names):
            if gt_id == pred_id:
                continue
            pixels = int(confusion[gt_id, pred_id])
            if pixels == 0:
                continue
            rows.append(
                {
                    "gt_class_id": gt_id,
                    "gt_class_name": gt_name,
                    "pred_class_id": pred_id,
                    "pred_class_name": pred_name,
                    "pixels": pixels,
                    "rate_within_gt_class": safe_ratio(pixels, int(gt_pixels[gt_id])),
                }
            )
    rows_sorted = sorted(rows, key=lambda r: int(r["pixels"]), reverse=True)
    write_csv(
        output_dir / f"top_confusion_pairs_{analysis_name}.csv",
        rows_sorted[:top_n],
        fieldnames=[
            "gt_class_id",
            "gt_class_name",
            "pred_class_id",
            "pred_class_name",
            "pixels",
            "rate_within_gt_class",
        ],
    )
    return rows_sorted[:top_n]


def plot_top_confusion_pairs(
    rows: list[dict[str, Any]],
    output_path: Path,
    analysis_label: str,
    top_n: int = 15,
) -> None:
    rows = rows[:top_n]
    if not rows:
        return
    labels = [f"{r['gt_class_name']} → {r['pred_class_name']}" for r in rows]
    values = np.array([float(r["pixels"]) for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified pixels")
    ax.set_title(f"Top confusion pairs ({analysis_label})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_error_rates(
    records: list[ImageErrorRecord],
    output_path: Path,
    analysis_label: str,
) -> None:
    if not records:
        return
    sorted_records = sorted(records, key=lambda r: r.pixel_error_rate, reverse=True)
    x = np.arange(len(sorted_records))
    pixel_error = np.array([r.pixel_error_rate for r in sorted_records], dtype=np.float64)
    boundary_error = np.array([r.boundary_error_rate for r in sorted_records], dtype=np.float64)
    interior_error = np.array([r.interior_error_rate for r in sorted_records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, pixel_error, label="overall pixel error")
    ax.plot(x, boundary_error, label="boundary error")
    ax.plot(x, interior_error, label="interior error")
    ax.set_xlabel(f"Images sorted by overall pixel error ({analysis_label})")
    ax.set_ylabel("Error rate")
    ax.set_title(f"Per-image error rates ({analysis_label})")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_background_rates(
    records: list[ImageErrorRecord],
    output_path: Path,
    analysis_label: str,
) -> None:
    valid_records = [r for r in records if not math.isnan(r.dynamic_to_background_error_rate)]
    if not valid_records:
        return
    sorted_records = sorted(
        valid_records,
        key=lambda r: r.dynamic_to_background_error_rate,
        reverse=True,
    )
    rates = np.array([r.dynamic_to_background_error_rate for r in sorted_records])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(rates)), rates)
    ax.set_xlabel(f"Images with dynamic GT pixels, sorted by rate ({analysis_label})")
    ax.set_ylabel("Dynamic → background rate")
    ax.set_title(f"Dynamic-object to background confusion per image ({analysis_label})")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_error_map(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    boundary_band: np.ndarray,
    dynamic_to_background: np.ndarray,
) -> np.ndarray:
    h, w = gt.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    wrong = (pred != gt) & valid
    other_error = wrong.copy()

    boundary_error = wrong & boundary_band
    dyn_bg_error = wrong & dynamic_to_background

    error_map[other_error] = ERROR_COLORS["other_error"]
    error_map[boundary_error] = ERROR_COLORS["boundary_error"]
    error_map[dyn_bg_error] = ERROR_COLORS["dynamic_to_background"]
    return error_map


def make_error_overlay(image: np.ndarray, error_map: np.ndarray, alpha: float) -> np.ndarray:
    has_error = error_map.sum(axis=2) > 0
    overlay = image.astype(np.float32).copy()
    overlay[has_error] = (
        (1.0 - alpha) * overlay[has_error] + alpha * error_map[has_error].astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_error_visualization(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    error_map: np.ndarray,
    output_path: Path,
    config: dict[str, Any],
    title: str,
    alpha: float,
    dpi: int,
) -> None:
    num_classes = int(config["data"].get("num_classes", 19))
    ignore_index = int(config["data"].get("ignore_index", 255))
    gt_color = colorize_mask(gt, num_classes=num_classes, ignore_index=ignore_index)
    pred_color = colorize_mask(pred, num_classes=num_classes, ignore_index=ignore_index)
    pred_overlay = make_overlay(image, pred_color, alpha=alpha)
    error_overlay = make_error_overlay(image, error_map, alpha=alpha)

    panels = [
        ("Image", image),
        ("GT", gt_color),
        ("Prediction", pred_color),
        ("Error Map\nred=boundary, yellow=dynamic→bg, blue=other", error_map),
        ("Error Overlay", error_overlay),
        ("Pred Overlay", pred_overlay),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (panel_title, panel_image) in zip(axes, panels):
        ax.imshow(panel_image)
        ax.set_title(panel_title, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_selected_examples(
    records_by_category: dict[str, list[ImageErrorRecord]],
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    condition: str,
    output_dir: Path,
    boundary_radius: int,
    alpha: float,
    dpi: int,
    device: torch.device,
) -> None:
    dataset = build_dataset(config, split=split)
    model = build_model(config).to(device)
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    class_names = get_class_names()
    num_classes = int(config["data"].get("num_classes", len(class_names)))
    ignore_index = int(config["data"].get("ignore_index", 255))
    dynamic_ids = ids_from_names(DYNAMIC_CLASSES, class_names)
    background_ids = ids_from_names(BACKGROUND_CLASSES, class_names)

    examples_manifest: list[dict[str, Any]] = []

    with torch.no_grad():
        for category, records in records_by_category.items():
            category_dir = ensure_dir(output_dir / "examples" / category)
            for rank, record in enumerate(records, start=1):
                sample = dataset[record.dataset_index]
                image_tensor = sample["image"]
                gt = sample["mask"].detach().cpu().numpy().astype(np.int64)
                image = denormalize_image(image_tensor, config)
                logits = model(image_tensor.unsqueeze(0).to(device, non_blocking=True))
                pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)

                valid = gt != ignore_index
                valid &= gt >= 0
                valid &= gt < num_classes
                valid &= pred >= 0
                valid &= pred < num_classes
                boundary_band = make_semantic_boundary_band(
                    gt=gt,
                    valid_mask=valid,
                    radius=boundary_radius,
                )
                dynamic_gt = np.isin(gt, dynamic_ids) & valid
                pred_background = np.isin(pred, background_ids) & valid
                dynamic_to_background = dynamic_gt & pred_background
                error_map = make_error_map(
                    pred=pred,
                    gt=gt,
                    valid=valid,
                    boundary_band=boundary_band,
                    dynamic_to_background=dynamic_to_background,
                )

                image_path = str(sample.get("image_path", f"sample_{record.dataset_index}"))
                filename = (
                    f"{rank:02d}_idx{record.dataset_index:05d}_"
                    f"miou{format_rate(record.image_miou)}_"
                    f"berr{format_rate(record.boundary_error_rate)}_"
                    f"dbg{format_rate(record.dynamic_to_background_error_rate)}_"
                    f"{safe_filename(image_path)}.png"
                )
                # Keep filenames filesystem-friendly.
                filename = re.sub(r"[^a-zA-Z0-9가-힣_.=-]+", "_", filename)
                output_path = category_dir / filename
                title = (
                    f"{category} | idx={record.dataset_index} | condition={condition} | "
                    f"mIoU={format_rate(record.image_miou)} | "
                    f"pixel_err={format_rate(record.pixel_error_rate)} | "
                    f"boundary_err={format_rate(record.boundary_error_rate)} | "
                    f"dynamic→bg={format_rate(record.dynamic_to_background_error_rate)}"
                )
                save_error_visualization(
                    image=image,
                    gt=gt,
                    pred=pred,
                    error_map=error_map,
                    output_path=output_path,
                    config=config,
                    title=title,
                    alpha=alpha,
                    dpi=dpi,
                )
                examples_manifest.append(
                    {
                        "category": category,
                        "rank": rank,
                        "dataset_index": record.dataset_index,
                        "image_path": image_path,
                        "output_path": str(output_path),
                        "image_miou": record.image_miou,
                        "pixel_error_rate": record.pixel_error_rate,
                        "boundary_error_rate": record.boundary_error_rate,
                        "dynamic_to_background_error_rate": record.dynamic_to_background_error_rate,
                    }
                )
                print(f"Saved example: {output_path}")

    write_csv(
        output_dir / "examples" / "examples_manifest.csv",
        examples_manifest,
        fieldnames=[
            "category",
            "rank",
            "dataset_index",
            "image_path",
            "output_path",
            "image_miou",
            "pixel_error_rate",
            "boundary_error_rate",
            "dynamic_to_background_error_rate",
        ],
    )


def select_example_records(
    records: list[ImageErrorRecord],
    top_k: int,
    min_dynamic_pixels: int,
) -> dict[str, list[ImageErrorRecord]]:
    finite_miou = [r for r in records if not math.isnan(r.image_miou)]
    worst_overall = sorted(finite_miou, key=lambda r: r.image_miou)[:top_k]
    best_cases = sorted(finite_miou, key=lambda r: r.image_miou, reverse=True)[:top_k]

    boundary_candidates = [
        r for r in records if r.boundary_pixels > 0 and not math.isnan(r.boundary_error_rate)
    ]
    boundary_error = sorted(
        boundary_candidates,
        key=lambda r: (r.boundary_error_rate, r.boundary_error_pixels),
        reverse=True,
    )[:top_k]

    dbg_candidates = [
        r
        for r in records
        if r.gt_dynamic_pixels >= min_dynamic_pixels
        and not math.isnan(r.dynamic_to_background_error_rate)
    ]
    dynamic_to_background = sorted(
        dbg_candidates,
        key=lambda r: (
            r.dynamic_to_background_error_rate,
            r.dynamic_to_background_error_pixels,
        ),
        reverse=True,
    )[:top_k]

    return {
        "worst_overall": worst_overall,
        "boundary_error": boundary_error,
        "dynamic_to_background": dynamic_to_background,
        "best_cases": best_cases,
    }


def main() -> None:
    args = parse_args()
    analyze_all = is_all_condition(args.condition)
    analysis_name = get_analysis_name(args.condition)
    analysis_label = get_analysis_label(args.condition)
    output_dir = ensure_dir(
        args.output_dir if args.output_dir is not None else Path("outputs") / "analysis" / analysis_name
    )
    plot_dir = ensure_dir(output_dir / "plots")

    config = load_config(args.config)
    class_names = get_class_names()
    checkpoint_path = resolve_checkpoint_path(args, config)
    device = torch.device(args.device) if args.device is not None else get_device()

    iou_drop_rows: list[dict[str, Any]] = []
    if analyze_all:
        print(
            "[1/4] Skipping class-wise IoU drop comparison "
            "because --condition none analyzes the whole split."
        )
    else:
        print("[1/4] Making class-wise IoU drop table and plots...")
        condition_result_path = (
            Path(args.condition_result)
            if args.condition_result is not None
            else Path("outputs") / "results" / analysis_name / "eval_val.json"
        )
        iou_drop_rows = make_class_iou_drop_outputs(
            overall_result_path=args.overall_result,
            condition_result_path=condition_result_path,
            class_names=class_names,
            output_dir=output_dir,
        )
        if iou_drop_rows:
            print("Top degraded classes by overall-val minus condition IoU:")
            for row in iou_drop_rows[:5]:
                print(
                    f"  {row['class_name']:<15} "
                    f"overall={float(row['overall_val_iou']):.4f}, "
                    f"condition={float(row['condition_iou']):.4f}, "
                    f"drop={float(row['iou_drop_overall_minus_condition']):.4f}"
                )

    print(f"[2/4] Running prediction-level error analysis ({analysis_label})...")
    records, confusion, extras = analyze_predictions(
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        condition=args.condition,
        output_dir=output_dir,
        boundary_radius=args.boundary_radius,
        max_samples=args.max_samples,
        device=device,
    )

    print("[3/4] Saving CSV summaries and plots...")
    record_rows = records_to_rows(records)
    image_summary_csv = output_dir / f"image_error_summary_{analysis_name}.csv"
    write_csv(
        image_summary_csv,
        record_rows,
        fieldnames=[
            "dataset_index",
            "condition",
            "image_path",
            "label_path",
            "image_miou",
            "pixel_error_rate",
            "boundary_error_rate",
            "interior_error_rate",
            "dynamic_to_background_error_rate",
            "valid_pixels",
            "boundary_pixels",
            "interior_pixels",
            "gt_dynamic_pixels",
            "error_pixels",
            "boundary_error_pixels",
            "interior_error_pixels",
            "dynamic_to_background_error_pixels",
        ],
    )

    save_confusion_matrix(confusion, class_names, output_dir, analysis_name)
    class_summary = make_class_error_summary(
        confusion=confusion,
        class_names=class_names,
        boundary_pixels_by_class=extras["boundary_pixels_by_class"],
        boundary_errors_by_class=extras["boundary_errors_by_class"],
        output_dir=output_dir,
        analysis_name=analysis_name,
    )
    top_confusions = make_top_confusion_pairs(confusion, class_names, output_dir, analysis_name)
    plot_top_confusion_pairs(
        top_confusions,
        plot_dir / f"top_confusion_pairs_{analysis_name}.png",
        analysis_label=analysis_label,
    )
    plot_image_error_rates(
        records,
        plot_dir / f"image_error_rates_{analysis_name}.png",
        analysis_label=analysis_label,
    )
    plot_dynamic_background_rates(
        records,
        plot_dir / f"dynamic_to_background_rates_{analysis_name}.png",
        analysis_label=analysis_label,
    )

    miou_from_predictions, class_iou_from_predictions = compute_iou_from_confusion(confusion)
    overall_error_rate = safe_ratio(sum(r.error_pixels for r in records), sum(r.valid_pixels for r in records))
    overall_boundary_error_rate = safe_ratio(
        sum(r.boundary_error_pixels for r in records),
        sum(r.boundary_pixels for r in records),
    )
    overall_interior_error_rate = safe_ratio(
        sum(r.interior_error_pixels for r in records),
        sum(r.interior_pixels for r in records),
    )
    overall_dynamic_to_background_rate = safe_ratio(
        sum(r.dynamic_to_background_error_pixels for r in records),
        sum(r.gt_dynamic_pixels for r in records),
    )

    summary = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "condition": args.condition,
        "analysis_name": analysis_name,
        "analysis_label": analysis_label,
        "num_images": len(records),
        "miou_from_predictions": miou_from_predictions,
        "overall_pixel_error_rate": overall_error_rate,
        "boundary_radius": args.boundary_radius,
        "boundary_error_rate": overall_boundary_error_rate,
        "interior_error_rate": overall_interior_error_rate,
        "dynamic_classes": DYNAMIC_CLASSES,
        "background_classes": BACKGROUND_CLASSES,
        "dynamic_to_background_error_rate": overall_dynamic_to_background_rate,
        "top_iou_drops": iou_drop_rows[:10],
        "top_confusion_pairs": top_confusions[:10],
        "class_error_summary_top": class_summary[:10],
        "outputs": {
            "image_error_summary_csv": str(image_summary_csv),
            "class_error_summary_csv": str(output_dir / f"class_error_summary_{analysis_name}.csv"),
            "confusion_matrix_csv": str(output_dir / f"confusion_matrix_{analysis_name}.csv"),
            "top_confusion_pairs_csv": str(output_dir / f"top_confusion_pairs_{analysis_name}.csv"),
            "plots_dir": str(plot_dir),
            "examples_dir": str(output_dir / "examples"),
        },
    }
    if not analyze_all:
        summary["outputs"]["class_iou_drop_csv"] = str(output_dir / "class_iou_drop.csv")

    summary_json = output_dir / f"{analysis_name}_error_analysis_summary.json"
    save_json(summary, summary_json)

    print("[4/4] Saving representative example images...")
    selected_examples = select_example_records(
        records=records,
        top_k=args.top_k,
        min_dynamic_pixels=args.min_dynamic_pixels,
    )
    save_selected_examples(
        records_by_category=selected_examples,
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        condition=analysis_label,
        output_dir=output_dir,
        boundary_radius=args.boundary_radius,
        alpha=args.alpha,
        dpi=args.dpi,
        device=device,
    )

    print("Done.")
    print(f"Summary JSON: {summary_json}")
    if not analyze_all:
        print(f"Class IoU drop CSV: {output_dir / 'class_iou_drop.csv'}")
    print(f"Per-image summary CSV: {image_summary_csv}")
    print(f"Examples: {output_dir / 'examples'}")


if __name__ == "__main__":
    main()
