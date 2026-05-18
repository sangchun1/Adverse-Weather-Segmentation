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
from awseg.utils import AverageMeter, format_metrics, get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size. Defaults to train.batch_size.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help=(
            "Optional weather condition filter. "
            "Examples: fog, rain, snow, night. "
            "If set, evaluation uses only this condition."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/baseline",
        help="Directory to save small JSON evaluation summaries for GitHub tracking.",
    )
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        default=True,
        help="Disable saving JSON evaluation summaries.",
    )
    return parser.parse_args()



def _json_safe(value: Any) -> Any:
    """Convert values to strict JSON-safe objects."""
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
    """Save dictionary as strict JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(data),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def get_condition_label(condition: str | None) -> str | None:
    """Return condition value for JSON content.

    None means all conditions are used.
    """
    return condition


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_available_conditions(dataset: Any) -> list[str]:
    """Return sorted condition names from a dataset created by build_dataset()."""
    if not hasattr(dataset, "samples"):
        return []

    return sorted(
        {str(sample.get("condition", "unknown")) for sample in dataset.samples}
    )


def filter_dataset_by_condition(dataset: Any, condition: str, split: str) -> Subset:
    """Filter dataset by weather condition using dataset.samples metadata."""
    if not hasattr(dataset, "samples"):
        raise AttributeError(
            "Dataset does not have `samples` metadata, so condition filtering is unavailable."
        )

    indices = [
        idx
        for idx, sample in enumerate(dataset.samples)
        if str(sample.get("condition", "unknown")) == condition
    ]

    if len(indices) == 0:
        available_conditions = get_available_conditions(dataset)
        raise ValueError(
            f"No samples found for condition={condition!r} in split={split!r}. "
            f"Available conditions: {available_conditions}"
        )

    print(
        f"Using condition filter for {split}: {condition} "
        f"({len(indices)} / {len(dataset)} samples)"
    )

    return Subset(dataset, indices)


def build_eval_dataloader(
    config: Dict[str, Any],
    split: str,
    batch_size: int | None = None,
    condition: str | None = None,
) -> DataLoader:
    dataset = build_dataset(config, split=split)

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

    # Fallback for raw state_dict checkpoints.
    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


@torch.no_grad()
def evaluate(
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
        images = batch["image"].to(device, non_blocking=True)

        if "mask" not in batch:
            raise ValueError(
                "This split does not contain labels, so mIoU cannot be computed. "
                "Use split='val' or provide labels for evaluation."
            )

        masks = batch["mask"].to(device, non_blocking=True)
        conditions = batch.get("condition", ["unknown"] * images.size(0))

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        total_metric.update(preds, masks)

        # Condition-wise mIoU is useful for adverse weather robustness analysis.
        for idx, condition in enumerate(conditions):
            condition = str(condition)

            if condition not in condition_metrics:
                condition_metrics[condition] = SegmentationMetric(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    device=device,
                )

            condition_metrics[condition].update(
                preds[idx : idx + 1],
                masks[idx : idx + 1],
            )
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


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"Using device: {device}")

    dataloader = build_eval_dataloader(
        config=config,
        split=args.split,
        batch_size=args.batch_size,
        condition=args.condition,
    )

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)

    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")

    print(f"Evaluating split: {args.split}")
    if args.condition is not None:
        print(f"Condition filter: {args.condition}")
    print(f"Samples: {len(dataloader.dataset)}")

    result = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        config=config,
    )

    class_names = get_class_names()

    print()
    print(format_metrics({"miou": result["miou"]}, prefix="Overall"))
    print()
    print("Class-wise IoU")
    print(format_class_iou(result["class_iou"], class_names=class_names))

    condition_results = result["condition_results"]

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

    if args.save_results:
        condition_label = get_condition_label(args.condition)
        result_path = Path(args.result_dir) / f"eval_{args.split}.json"

        class_iou = {
            class_name: iou
            for class_name, iou in zip(class_names, result["class_iou"])
        }

        evaluation_summary = {
            "task": "evaluate",
            "created_at": get_timestamp(),
            "config_path": str(args.config),
            "checkpoint": str(args.checkpoint),
            "split": str(args.split),
            "condition": condition_label,
            "num_samples": int(len(dataloader.dataset)),
            "miou": float(result["miou"]),
            "class_iou": class_iou,
            "condition_results": condition_results,
        }

        save_json(evaluation_summary, result_path)
        print()
        print(f"Saved evaluation result JSON: {result_path}")


if __name__ == "__main__":
    main()
