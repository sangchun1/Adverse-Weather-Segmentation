#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from awseg.dataset import build_dataset  # noqa: E402
from awseg.models import build_model  # noqa: E402

try:
    from awseg.dataset import get_class_names as _get_class_names  # noqa: E402
except ImportError:  # pragma: no cover
    _get_class_names = None

try:
    from awseg.utils import get_device as _get_device  # noqa: E402
except ImportError:  # pragma: no cover
    _get_device = None

try:
    from awseg.utils import load_config as _load_config  # noqa: E402
except ImportError:  # pragma: no cover
    _load_config = None


ACDC_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
DYNAMIC_CLASSES = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
BACKGROUND_CLASSES = {
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky",
}
ALL_CONDITION_VALUES = {"", "none", "all", "overall", "val", "baseline"}
SUPPORTED_GROUPS = {"baseline", "loss", "model", "augmentation", "enhancement", "proposed", "final"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze semantic segmentation errors for a whole split or one condition."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--group", type=str, default=None, choices=sorted(SUPPORTED_GROUPS))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--condition", type=str, default="none")
    parser.add_argument("--result-root", type=str, default="outputs/results")
    parser.add_argument("--checkpoint-root", type=str, default="outputs/checkpoints")
    parser.add_argument("--analysis-root", type=str, default="outputs/analysis")
    parser.add_argument("--overall-result", type=str, default=None)
    parser.add_argument("--condition-result", "--night-result", dest="condition_result", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--min-dynamic-pixels", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = str(value).strip().lower()
    return re.sub(r"[^a-zA-Z0-9_.=-]+", "_", value).strip("_")


def is_all_condition(condition: Optional[str]) -> bool:
    if condition is None:
        return True
    return str(condition).strip().lower() in ALL_CONDITION_VALUES


def infer_experiment_name(group: Optional[str], experiment: Optional[str]) -> str:
    if experiment:
        return sanitize_name(experiment)
    if group:
        return sanitize_name(group)
    return "analysis"


def infer_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return Path(args.config)
    group = args.group or "baseline"
    if group in {"baseline", "proposed", "final"}:
        return Path("configs") / f"{group}.yaml"
    if args.experiment:
        return Path("configs") / group / f"{args.experiment}.yaml"
    return Path("configs") / f"{group}.yaml"


def infer_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    name = infer_experiment_name(args.group, args.experiment)
    return Path(args.checkpoint_root) / name / "best_miou.pth"


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    name = infer_experiment_name(args.group, args.experiment)
    if not is_all_condition(args.condition):
        name = f"{name}_{sanitize_name(args.condition)}"
    return Path(args.analysis_root) / name


def load_config(path: Path) -> dict[str, Any]:
    if _load_config is not None:
        return _load_config(str(path))
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_class_names() -> list[str]:
    if _get_class_names is not None:
        return list(_get_class_names())
    return ACDC_CLASS_NAMES.copy()


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if _get_device is not None:
        return _get_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


def filter_dataset(dataset: Any, condition: Optional[str], max_samples: Optional[int]) -> Any:
    indices = list(range(len(dataset)))
    if not is_all_condition(condition):
        target = str(condition).strip().lower()
        indices = [
            idx for idx, sample in enumerate(dataset.samples)
            if str(sample.get("condition", "unknown")).strip().lower() == target
        ]
        if not indices:
            available = sorted({str(sample.get("condition", "unknown")) for sample in dataset.samples})
            raise ValueError(f"No samples for condition={condition!r}. Available conditions: {available}")
    if max_samples is not None:
        indices = indices[:max_samples]
    return Subset(dataset, indices)


def make_dataloader(config: dict[str, Any], split: str, condition: Optional[str], max_samples: Optional[int]) -> DataLoader:
    dataset = build_dataset(config, split=split)
    dataset = filter_dataset(dataset, condition=condition, max_samples=max_samples)
    eval_config = config.get("evaluate", {})
    train_config = config.get("train", {})
    batch_size = int(eval_config.get("batch_size", train_config.get("batch_size", 4)))
    num_workers = int(eval_config.get("num_workers", train_config.get("num_workers", 4)))
    pin_memory = bool(eval_config.get("pin_memory", train_config.get("pin_memory", torch.cuda.is_available())))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)


def safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def image_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> float:
    valid = target != ignore_index
    ious: list[float] = []
    for class_id in range(num_classes):
        pred_c = (pred == class_id) & valid
        target_c = (target == class_id) & valid
        union = pred_c | target_c
        union_count = int(union.sum().item())
        if union_count == 0:
            continue
        inter_count = int((pred_c & target_c).sum().item())
        ious.append(inter_count / max(union_count, 1))
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> None:
    valid = target != ignore_index
    encoded = target[valid].to(torch.int64) * num_classes + pred[valid].to(torch.int64)
    bins = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bins.reshape(num_classes, num_classes).to(confusion.device)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_csv(path: Path, confusion: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *class_names])
        for idx, row in enumerate(confusion):
            writer.writerow([class_names[idx], *[int(v) for v in row]])


def save_top_confusion_plot(path: Path, rows: list[dict[str, Any]], dpi: int) -> None:
    if not rows:
        return
    labels = [f"{r['gt_class']}→{r['pred_class']}" for r in rows]
    values = [float(r["count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.45)))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("Pixels")
    ax.set_title("Top confusion pairs")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_eval_json(path: Optional[str]) -> Optional[dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_class_iou(summary: dict[str, Any]) -> Optional[dict[str, float]]:
    if "main" in summary and isinstance(summary["main"], dict):
        return summary["main"].get("class_iou")
    return summary.get("class_iou")


def save_class_iou_drop(args: argparse.Namespace, output_dir: Path) -> None:
    overall = load_eval_json(args.overall_result)
    condition = load_eval_json(args.condition_result)
    if overall is None or condition is None:
        return
    overall_iou = extract_class_iou(overall)
    condition_iou = extract_class_iou(condition)
    if not overall_iou or not condition_iou:
        return
    rows = []
    for name, overall_value in overall_iou.items():
        cond_value = condition_iou.get(name)
        if cond_value is None:
            continue
        rows.append(
            {
                "class": name,
                "overall_iou": overall_value,
                "condition_iou": cond_value,
                "condition_minus_overall": float(cond_value) - float(overall_value),
            }
        )
    write_csv(output_dir / "class_iou_drop.csv", rows, ["class", "overall_iou", "condition_iou", "condition_minus_overall"])


def main() -> None:
    args = parse_args()
    config_path = infer_config_path(args)
    checkpoint_path = infer_checkpoint_path(args)
    output_dir = infer_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    class_names = get_class_names()
    num_classes = int(config["data"]["num_classes"])
    ignore_index = int(config["data"].get("ignore_index", 255))
    device = get_device(args.device)

    print(f"Using device: {device}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")
    model.eval()

    dataloader = make_dataloader(config, split=args.split, condition=args.condition, max_samples=args.max_samples)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    records: list[dict[str, Any]] = []
    dynamic_ids = {idx for idx, name in enumerate(class_names) if name in DYNAMIC_CLASSES}
    background_ids = {idx for idx, name in enumerate(class_names) if name in BACKGROUND_CLASSES}

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc="Error analysis",
        dynamic_ncols=True,
        mininterval=5,
        file=sys.stdout,
        leave=True,
    )

    with torch.no_grad():
        seen = 0
        for batch in progress_bar:
            if "mask" not in batch:
                raise ValueError("This split has no labels, so error analysis cannot be computed.")
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            update_confusion(confusion, preds, masks, num_classes=num_classes, ignore_index=ignore_index)

            batch_size = images.size(0)
            for i in range(batch_size):
                pred = preds[i]
                target = masks[i]
                valid = target != ignore_index
                valid_pixels = int(valid.sum().item())
                errors = (pred != target) & valid
                error_pixels = int(errors.sum().item())
                gt_dynamic = torch.zeros_like(valid, dtype=torch.bool)
                pred_background = torch.zeros_like(valid, dtype=torch.bool)
                for class_id in dynamic_ids:
                    gt_dynamic |= target == class_id
                for class_id in background_ids:
                    pred_background |= pred == class_id
                dynamic_to_background = errors & gt_dynamic & pred_background
                image_path = batch.get("image_path", [""] * batch_size)[i]
                label_path = batch.get("label_path", [""] * batch_size)[i] if "label_path" in batch else ""
                condition = batch.get("condition", ["unknown"] * batch_size)[i]
                records.append(
                    {
                        "dataset_index": seen + i,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "condition": str(condition),
                        "image_miou": safe_float(image_miou(pred, target, num_classes, ignore_index)),
                        "pixel_error_rate": safe_float(error_pixels / max(valid_pixels, 1)),
                        "boundary_error_rate": 0.0,
                        "interior_error_rate": safe_float(error_pixels / max(valid_pixels, 1)),
                        "dynamic_to_background_error_rate": safe_float(int(dynamic_to_background.sum().item()) / max(int(gt_dynamic.sum().item()), 1)),
                        "valid_pixels": valid_pixels,
                        "boundary_pixels": 0,
                        "interior_pixels": valid_pixels,
                        "gt_dynamic_pixels": int(gt_dynamic.sum().item()),
                        "error_pixels": error_pixels,
                        "boundary_error_pixels": 0,
                        "interior_error_pixels": error_pixels,
                        "dynamic_to_background_error_pixels": int(dynamic_to_background.sum().item()),
                    }
                )
            seen += batch_size
            progress_bar.set_postfix(images=seen)

    fieldnames = [
        "dataset_index", "image_path", "label_path", "condition", "image_miou",
        "pixel_error_rate", "boundary_error_rate", "interior_error_rate",
        "dynamic_to_background_error_rate", "valid_pixels", "boundary_pixels",
        "interior_pixels", "gt_dynamic_pixels", "error_pixels", "boundary_error_pixels",
        "interior_error_pixels", "dynamic_to_background_error_pixels",
    ]
    write_csv(output_dir / "per_image_error_summary.csv", records, fieldnames)

    confusion_np = confusion.detach().cpu().numpy()
    save_confusion_csv(output_dir / "confusion_matrix.csv", confusion_np, class_names)

    class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        tp = int(confusion_np[class_id, class_id])
        fn = int(confusion_np[class_id, :].sum() - tp)
        fp = int(confusion_np[:, class_id].sum() - tp)
        union = tp + fp + fn
        class_rows.append(
            {
                "class_id": class_id,
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": safe_float(tp / max(union, 1)),
                "error_pixels": fp + fn,
            }
        )
    write_csv(output_dir / "class_error_summary.csv", class_rows, ["class_id", "class", "tp", "fp", "fn", "iou", "error_pixels"])

    pairs: list[dict[str, Any]] = []
    for gt_id in range(num_classes):
        for pred_id in range(num_classes):
            if gt_id == pred_id:
                continue
            count = int(confusion_np[gt_id, pred_id])
            if count <= 0:
                continue
            pairs.append({"gt_class": class_names[gt_id], "pred_class": class_names[pred_id], "count": count})
    pairs.sort(key=lambda row: int(row["count"]), reverse=True)
    top_pairs = pairs[: max(args.top_k * 3, args.top_k)]
    write_csv(output_dir / "top_confusion_pairs.csv", top_pairs, ["gt_class", "pred_class", "count"])
    save_top_confusion_plot(output_dir / "top_confusion_pairs.png", top_pairs[:20], dpi=args.dpi)
    save_class_iou_drop(args, output_dir)

    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "condition": args.condition,
        "num_images": len(records),
        "output_dir": str(output_dir),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved error analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
