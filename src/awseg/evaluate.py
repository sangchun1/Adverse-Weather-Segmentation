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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", help="Main split: train, val, or test.")
    parser.add_argument("--condition", type=str, default=None, help="Optional filter: fog, rain, snow, night.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--normal-split", type=str, default="normal")
    parser.add_argument("--include-normal", dest="include_normal", action="store_true", default=True)
    parser.add_argument("--no-include-normal", dest="include_normal", action="store_false")
    parser.add_argument("--result-dir", type=str, default="outputs/results/baseline")
    parser.add_argument("--result-path", type=str, default=None)
    parser.add_argument("--no-save-results", dest="save_results", action="store_false", default=True)
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0, cuda:1, or cpu.")
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
) -> Subset:
    indices = []

    for idx, sample in enumerate(dataset.samples):
        if str(sample.get("split", "")) != target_split:
            continue

        if require_label and not str(sample.get("label_path", "")).strip():
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
) -> DataLoader:
    dataset = build_dataset(config, split=split)

    if csv_split_filter is not None:
        dataset = filter_dataset_by_split_column(dataset, target_split=csv_split_filter)

    if condition is not None:
        dataset = filter_dataset_by_condition(dataset, condition=condition, split=split)

    eval_config = config.get("evaluate", {})
    train_config = config.get("train", {})

    if batch_size is None:
        batch_size = int(eval_config.get("batch_size", train_config.get("batch_size", 4)))

    num_workers = int(eval_config.get("num_workers", train_config.get("num_workers", 4)))
    pin_memory = bool(eval_config.get("pin_memory", train_config.get("pin_memory", torch.cuda.is_available())))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
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
) -> Dict[str, Any]:
    class_iou = {
        class_name: iou
        for class_name, iou in zip(class_names, result["class_iou"])
    }

    return {
        "split": split,
        "condition": condition,
        "source_csv": source_csv,
        "csv_split_filter": csv_split_filter,
        "num_samples": int(len(dataloader.dataset)),
        "miou": float(result["miou"]),
        "class_iou": class_iou,
        "condition_results": result["condition_results"],
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


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = torch.device(args.device) if args.device is not None else get_device()

    print(f"Using device: {device}")

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)

    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")

    class_names = get_class_names()

    main_loader = build_eval_dataloader(
        config=config,
        split=args.split,
        batch_size=args.batch_size,
        condition=args.condition,
        csv_split_filter=None,
    )

    print(f"Evaluating main split: {args.split}")

    if args.condition is not None:
        print(f"Condition filter: {args.condition}")

    print(f"Samples: {len(main_loader.dataset)}")

    main_result = evaluate_split(model=model, dataloader=main_loader, device=device, config=config)
    main_summary = make_split_summary(
        split=args.split,
        condition=args.condition,
        dataloader=main_loader,
        result=main_result,
        class_names=class_names,
        source_csv=f"{args.split}.csv",
    )
    print_split_result("Main evaluation", main_summary, class_names)

    normal_summary = None

    if args.include_normal and split_csv_exists(config, args.normal_split):
        normal_loader = build_eval_dataloader(
            config=config,
            split=args.normal_split,
            batch_size=args.batch_size,
            condition=None,
            csv_split_filter=args.split,
        )

        print()
        print(f"Evaluating normal comparison from {args.normal_split}.csv")
        print(f"Normal CSV split filter: {args.split}")
        print(f"Samples: {len(normal_loader.dataset)}")

        normal_result = evaluate_split(
            model=model,
            dataloader=normal_loader,
            device=device,
            config=config,
        )
        normal_summary = make_split_summary(
            split=args.normal_split,
            condition=None,
            dataloader=normal_loader,
            result=normal_result,
            class_names=class_names,
            source_csv=f"{args.normal_split}.csv",
            csv_split_filter=args.split,
        )
        print_split_result("Normal-condition evaluation", normal_summary, class_names)

    elif args.include_normal:
        print(f"[WARN] data/splits/{args.normal_split}.csv not found. Skipping normal evaluation.")

    comparison = None
    if normal_summary is not None:
        comparison = {
            "normal_minus_main_miou": float(normal_summary["miou"] - main_summary["miou"]),
            "main_minus_normal_miou": float(main_summary["miou"] - normal_summary["miou"]),
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
