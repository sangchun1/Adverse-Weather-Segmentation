from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset

from awseg.dataset import build_dataset, get_class_names
from awseg.metrics import SegmentationMetric, format_class_iou
from awseg.models import build_model
from awseg.utils import format_metrics, get_device, load_config


CONDITIONS = ["fog", "rain", "snow", "night"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", help="Main split: train, val, or test.")
    parser.add_argument("--condition", type=str, default=None, help="Optional filter: fog, rain, snow, night.")
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Evaluate overall and every weather condition into one JSON file.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--normal-split", type=str, default="normal")
    parser.add_argument("--include-normal", dest="include_normal", action="store_true", default=True)
    parser.add_argument("--no-include-normal", dest="include_normal", action="store_false")
    parser.add_argument("--result-dir", type=str, default="outputs/results/baseline")
    parser.add_argument("--result-path", type=str, default=None)
    parser.add_argument("--no-save-results", dest="save_results", action="store_false", default=True)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, Path):
        return str(value)

    return value


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False, allow_nan=False)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_result_path(result_dir: str | Path, split: str, result_path: str | None) -> Path:
    if result_path is not None:
        return Path(result_path)

    return Path(result_dir) / f"eval_{split}.json"


def split_csv_exists(config: Dict[str, Any], split: str) -> bool:
    data_config = config["data"]
    root = Path(data_config.get("root", "."))
    split_dir = Path(data_config.get("split_dir", "data/splits"))

    if not split_dir.is_absolute():
        split_dir = root / split_dir

    return (split_dir / f"{split}.csv").exists()


def get_available_conditions(dataset: Any) -> list[str]:
    if not hasattr(dataset, "samples"):
        return []

    return sorted({str(sample.get("condition", "unknown")) for sample in dataset.samples})


def filter_dataset_by_condition(dataset: Any, condition: str, split: str) -> Subset:
    indices = [
        idx
        for idx, sample in enumerate(dataset.samples)
        if str(sample.get("condition", "unknown")) == condition
    ]

    if len(indices) == 0:
        raise ValueError(
            f"No samples found for condition={condition!r} in split={split!r}. "
            f"Available conditions: {get_available_conditions(dataset)}"
        )

    print(f"Using condition filter for {split}: {condition} ({len(indices)} / {len(dataset)} samples)")
    return Subset(dataset, indices)


def filter_dataset_by_split_column(
    dataset: Any,
    target_split: str,
    require_label: bool = True,
    condition: str | None = None,
) -> Subset:
    indices = []

    for idx, sample in enumerate(dataset.samples):
        if str(sample.get("split", "")) != target_split:
            continue

        if require_label and not str(sample.get("label_path", "")).strip():
            continue

        if condition is not None:
            image_path = str(sample.get("image_path", "")).replace("\\", "/")
            label_path = str(sample.get("label_path", "")).replace("\\", "/")
            token = f"/{condition}/"

            if token not in image_path and token not in label_path:
                continue

        indices.append(idx)

    if len(indices) == 0:
        available_splits = sorted({str(sample.get("split", "")) for sample in dataset.samples})
        raise ValueError(
            f"No normal samples found for split={target_split!r}. "
            f"Available split values in normal.csv: {available_splits}"
        )

    print(f"Using normal split filter: split={target_split} ({len(indices)} / {len(dataset)} samples)")
    return Subset(dataset, indices)


def build_eval_dataloader(
    config: Dict[str, Any],
    split: str,
    batch_size: int | None = None,
    condition: str | None = None,
    csv_split_filter: str | None = None,
    normal_condition: str | None = None,
) -> DataLoader:
    dataset = build_dataset(config, split=split)

    if csv_split_filter is not None:
        dataset = filter_dataset_by_split_column(
            dataset,
            target_split=csv_split_filter,
            condition=normal_condition,
        )

    if condition is not None:
        dataset = filter_dataset_by_condition(dataset, condition=condition, split=split)

    if batch_size is None:
        batch_size = int(config["train"]["batch_size"])

    num_workers = int(config["train"].get("num_workers", 4))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


def infer_ref_condition(path: str) -> str:
    normalized_path = str(path).replace("\\", "/")

    for condition in CONDITIONS:
        if f"/{condition}/" in normalized_path:
            return f"{condition}_ref"

    return "normal_ref"


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    normal_ref_conditions: bool = False,
) -> Dict[str, Any]:
    model.eval()

    num_classes = int(config["data"]["num_classes"])
    ignore_index = int(config["data"].get("ignore_index", 255))

    total_metric = SegmentationMetric(
        num_classes=num_classes,
        ignore_index=ignore_index,
        device=device,
    )
    total_metric.reset()

    condition_metrics: dict[str, SegmentationMetric] = {}
    num_images_by_condition: dict[str, int] = defaultdict(int)

    for batch in dataloader:
        if "mask" not in batch:
            raise ValueError("This split has no labels, so mIoU cannot be computed.")

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        conditions = batch.get("condition", ["unknown"] * images.size(0))

        if normal_ref_conditions:
            image_paths = batch.get("image_path", [""] * images.size(0))
            conditions = [infer_ref_condition(path) for path in image_paths]

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        total_metric.update(preds, masks)

        for idx, condition in enumerate(conditions):
            condition = str(condition)

            if condition not in condition_metrics:
                condition_metrics[condition] = SegmentationMetric(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    device=device,
                )

            condition_metrics[condition].update(preds[idx : idx + 1], masks[idx : idx + 1])
            num_images_by_condition[condition] += 1

    total_result = total_metric.compute()

    condition_results = {}
    for condition, metric in condition_metrics.items():
        result = metric.compute()
        condition_results[condition] = {
            "miou": float(result["miou"]),
            "class_iou": result["class_iou"],
            "num_images": int(num_images_by_condition[condition]),
        }

    return {
        "miou": float(total_result["miou"]),
        "class_iou": total_result["class_iou"],
        "condition_results": condition_results,
    }


def make_split_summary(
    split: str,
    condition: str | None,
    dataloader: DataLoader,
    result: Dict[str, Any],
    class_names: list[str],
    source_csv: str,
    csv_split_filter: str | None = None,
    is_normal_ref: bool = False,
) -> Dict[str, Any]:
    class_iou = {
        class_name: iou
        for class_name, iou in zip(class_names, result["class_iou"])
    }

    summary_condition = condition
    condition_results = result["condition_results"]

    if is_normal_ref:
        if condition is not None:
            summary_condition = f"{condition}_ref"

            if set(condition_results.keys()) == {"normal"}:
                condition_results = {
                    summary_condition: condition_results["normal"],
                }

    return {
        "split": split,
        "condition": summary_condition,
        "source_csv": source_csv,
        "csv_split_filter": csv_split_filter,
        "num_samples": int(len(dataloader.dataset)),
        "miou": float(result["miou"]),
        "class_iou": class_iou,
        "condition_results": condition_results,
    }


def make_comparison(
    main_summary: Dict[str, Any],
    normal_summary: Dict[str, Any] | None,
) -> Dict[str, float] | None:
    if normal_summary is None:
        return None

    return {
        "normal_minus_main_miou": float(normal_summary["miou"] - main_summary["miou"]),
        "main_minus_normal_miou": float(main_summary["miou"] - normal_summary["miou"]),
    }


def print_split_result(title: str, summary: Dict[str, Any], class_names: list[str]) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(
        f"Split: {summary['split']} | "
        f"Condition: {summary['condition']} | "
        f"Samples: {summary['num_samples']}"
    )

    if summary.get("csv_split_filter") is not None:
        print(f"CSV split filter: {summary['csv_split_filter']}")

    print(format_metrics({"miou": summary["miou"]}, prefix="Overall"))

    print()
    print("Class-wise IoU")
    class_iou_list = [summary["class_iou"][name] for name in class_names]
    print(format_class_iou(class_iou_list, class_names=class_names))

    condition_results = summary["condition_results"]

    if condition_results:
        print()
        print("Condition-wise mIoU")
        for condition in sorted(condition_results.keys()):
            condition_result = condition_results[condition]
            print(
                f"{condition}: "
                f"mIoU={condition_result['miou']:.4f}, "
                f"num_images={condition_result['num_images']}"
            )


def evaluate_pair(
    model: torch.nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    class_names: list[str],
    split: str,
    batch_size: int | None,
    condition: str | None,
    normal_split: str,
    include_normal: bool,
) -> tuple[Dict[str, Any], Dict[str, Any] | None, Dict[str, float] | None]:
    main_loader = build_eval_dataloader(
        config=config,
        split=split,
        batch_size=batch_size,
        condition=condition,
        csv_split_filter=None,
    )

    print(f"Evaluating main split: {split}")

    if condition is not None:
        print(f"Condition filter: {condition}")

    print(f"Samples: {len(main_loader.dataset)}")

    main_result = evaluate_split(
        model=model,
        dataloader=main_loader,
        device=device,
        config=config,
    )
    main_summary = make_split_summary(
        split=split,
        condition=condition,
        dataloader=main_loader,
        result=main_result,
        class_names=class_names,
        source_csv=f"{split}.csv",
    )
    print_split_result("Main evaluation", main_summary, class_names)

    normal_summary = None

    if include_normal and split_csv_exists(config, normal_split):
        normal_loader = build_eval_dataloader(
            config=config,
            split=normal_split,
            batch_size=batch_size,
            condition=None,
            csv_split_filter=split,
            normal_condition=condition,
        )

        print()
        print(f"Evaluating normal comparison from {normal_split}.csv")
        print(f"Normal CSV split filter: {split}")

        if condition is not None:
            print(f"Normal condition filter: {condition}_ref")

        print(f"Samples: {len(normal_loader.dataset)}")

        normal_result = evaluate_split(
            model=model,
            dataloader=normal_loader,
            device=device,
            config=config,
            normal_ref_conditions=True,
        )
        normal_summary = make_split_summary(
            split=normal_split,
            condition=condition,
            dataloader=normal_loader,
            result=normal_result,
            class_names=class_names,
            source_csv=f"{normal_split}.csv",
            csv_split_filter=split,
            is_normal_ref=True,
        )
        print_split_result("Normal-condition evaluation", normal_summary, class_names)

    elif include_normal:
        print(f"[WARN] data/splits/{normal_split}.csv not found. Skipping normal evaluation.")

    comparison = make_comparison(main_summary, normal_summary)
    return main_summary, normal_summary, comparison


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"Using device: {device}")

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)

    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")

    class_names = get_class_names()

    top_level_condition = None if args.all_conditions else args.condition

    main_summary, normal_summary, comparison = evaluate_pair(
        model=model,
        device=device,
        config=config,
        class_names=class_names,
        split=args.split,
        batch_size=args.batch_size,
        condition=top_level_condition,
        normal_split=args.normal_split,
        include_normal=args.include_normal,
    )

    per_condition = None

    if args.all_conditions:
        per_condition = {}

        for condition in CONDITIONS:
            print()
            print(f"Running per-condition evaluation: {condition}")
            condition_main, condition_normal, condition_comparison = evaluate_pair(
                model=model,
                device=device,
                config=config,
                class_names=class_names,
                split=args.split,
                batch_size=args.batch_size,
                condition=condition,
                normal_split=args.normal_split,
                include_normal=args.include_normal,
            )
            per_condition[condition] = {
                "main": condition_main,
                "normal": condition_normal,
                "comparison": condition_comparison,
            }

    output = {
        "task": "evaluate",
        "created_at": get_timestamp(),
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "checkpoint_best_miou": checkpoint.get("best_miou", None),
        "main": main_summary,
        "normal": normal_summary,
        "comparison": comparison,
    }

    if per_condition is not None:
        output["per_condition"] = per_condition

    if args.save_results:
        result_path = get_result_path(
            result_dir=args.result_dir,
            split=args.split,
            result_path=args.result_path,
        )
        save_json(output, result_path)
        print()
        print(f"Saved combined evaluation JSON: {result_path}")


if __name__ == "__main__":
    main()
