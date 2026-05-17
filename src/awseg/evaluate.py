from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

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
    return parser.parse_args()


def build_eval_dataloader(
    config: Dict[str, Any],
    split: str,
    batch_size: int | None = None,
) -> DataLoader:
    dataset = build_dataset(config, split=split)

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
    )

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)

    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")

    print(f"Evaluating split: {args.split}")
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


if __name__ == "__main__":
    main()
