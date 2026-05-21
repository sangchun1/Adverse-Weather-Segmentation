from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from awseg.dataset import build_dataset
from awseg.models import build_model
from awseg.utils import ensure_dir, get_device, load_config


CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128], [244, 35, 232], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35],
        [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize semantic segmentation predictions.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", help="Main split. Examples: train, val, test.")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations/baseline")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--condition", type=str, default=None, help="fog, rain, snow, night, normal, or None.")
    parser.add_argument("--samples-per-condition", type=int, default=5)
    parser.add_argument("--normal-split", type=str, default="normal")
    parser.add_argument("--include-normal", dest="include_normal", action="store_true", default=True)
    parser.add_argument("--no-include-normal", dest="include_normal", action="store_false")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def split_csv_exists(config: Dict[str, Any], split: str) -> bool:
    data_config = config["data"]
    root = Path(data_config.get("root", "."))
    split_dir = Path(data_config.get("split_dir", "data/splits"))

    if not split_dir.is_absolute():
        split_dir = root / split_dir

    return (split_dir / f"{split}.csv").exists()


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


def denormalize_image(image_tensor: torch.Tensor, config: Dict[str, Any]) -> np.ndarray:
    mean = np.array(config["data"].get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(config["data"].get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    image = image_tensor.detach().cpu().float().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)

    return (image * 255.0).astype(np.uint8)


def colorize_mask(mask: np.ndarray, num_classes: int = 19, ignore_index: int = 255) -> np.ndarray:
    mask = mask.astype(np.int64)
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id in range(num_classes):
        color_mask[mask == class_id] = CITYSCAPES_PALETTE[class_id]

    color_mask[mask == ignore_index] = np.array([0, 0, 0], dtype=np.uint8)
    return color_mask


def make_overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = (1.0 - alpha) * image.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def safe_filename(text: str) -> str:
    text = Path(text).stem
    text = re.sub(r"[^a-zA-Z0-9가-힣_.-]+", "_", text)
    return text[:120]


def save_visualization(
    image: np.ndarray,
    pred_mask: np.ndarray,
    output_path: Path,
    config: Dict[str, Any],
    gt_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    alpha: float = 0.55,
    dpi: int = 150,
) -> None:
    num_classes = int(config["data"]["num_classes"])
    ignore_index = int(config["data"].get("ignore_index", 255))

    pred_color = colorize_mask(pred_mask, num_classes=num_classes, ignore_index=ignore_index)
    pred_overlay = make_overlay(image, pred_color, alpha=alpha)

    if gt_mask is not None:
        gt_color = colorize_mask(gt_mask, num_classes=num_classes, ignore_index=ignore_index)
        panels = [("Image", image), ("GT", gt_color), ("Prediction", pred_color), ("Overlay", pred_overlay)]
    else:
        panels = [("Image", image), ("Prediction", pred_color), ("Overlay", pred_overlay)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))

    if len(panels) == 1:
        axes = [axes]

    for ax, (panel_title, panel_image) in zip(axes, panels):
        ax.imshow(panel_image)
        ax.set_title(panel_title)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_records(config: Dict[str, Any], split: str, include_normal: bool, normal_split: str) -> list[dict[str, Any]]:
    """Build records from main split and optional normal rows.

    normal.csv contains rows for train/val/test together, so this function keeps
    only rows where normal.csv's split column matches the requested split.
    """
    records: list[dict[str, Any]] = []

    main_dataset = build_dataset(config, split=split)

    for idx, sample in enumerate(main_dataset.samples):
        records.append(
            {
                "dataset": main_dataset,
                "index": idx,
                "condition": str(sample.get("condition", "unknown")),
                "csv_split": str(sample.get("split", split)),
                "source": split,
            }
        )

    if include_normal and split_csv_exists(config, normal_split):
        normal_dataset = build_dataset(config, split=normal_split)

        for idx, sample in enumerate(normal_dataset.samples):
            if str(sample.get("split", "")) != split:
                continue

            records.append(
                {
                    "dataset": normal_dataset,
                    "index": idx,
                    "condition": str(sample.get("condition", "normal")),
                    "csv_split": str(sample.get("split", split)),
                    "source": normal_split,
                }
            )

    elif include_normal:
        print(f"[WARN] data/splits/{normal_split}.csv not found. Skipping normal samples.")

    return records


def select_records(
    records: list[dict[str, Any]],
    num_samples: int,
    start_index: int,
    condition: Optional[str],
    samples_per_condition: Optional[int],
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    if condition is not None:
        filtered = [record for record in records if record["condition"] == condition]

        if shuffle:
            rng.shuffle(filtered)

        n = samples_per_condition if samples_per_condition is not None else num_samples
        return filtered[:n]

    if samples_per_condition is not None:
        records_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            records_by_condition[record["condition"]].append(record)

        selected: list[dict[str, Any]] = []
        preferred_order = ["fog", "rain", "snow", "night", "normal"]
        remaining = [
            name for name in sorted(records_by_condition.keys())
            if name not in preferred_order
        ]

        for condition_name in preferred_order + remaining:
            if condition_name not in records_by_condition:
                continue

            condition_records = records_by_condition[condition_name].copy()

            if shuffle:
                rng.shuffle(condition_records)

            selected.extend(condition_records[:samples_per_condition])

        return selected

    selected = records.copy()

    if shuffle:
        rng.shuffle(selected)
        return selected[:num_samples]

    end_index = min(start_index + num_samples, len(selected))
    return selected[start_index:end_index]


@torch.no_grad()
def visualize_predictions(
    config: Dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path,
    num_samples: int,
    start_index: int,
    condition: Optional[str],
    samples_per_condition: Optional[int],
    include_normal: bool,
    normal_split: str,
    shuffle: bool,
    seed: int,
    alpha: float,
    dpi: int,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    records = build_records(
        config=config,
        split=split,
        include_normal=include_normal,
        normal_split=normal_split,
    )

    selected_records = select_records(
        records=records,
        num_samples=num_samples,
        start_index=start_index,
        condition=condition,
        samples_per_condition=samples_per_condition,
        shuffle=shuffle,
        seed=seed,
    )

    if len(selected_records) == 0:
        available_conditions = sorted({record["condition"] for record in records})
        raise ValueError(
            f"No samples selected. condition={condition}. "
            f"Available conditions: {available_conditions}"
        )

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)

    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")

    model.eval()

    output_dir = ensure_dir(output_dir)

    condition_count: dict[str, int] = defaultdict(int)
    for record in selected_records:
        condition_count[record["condition"]] += 1

    print(f"Selected {len(selected_records)} samples")
    print("Selected samples by condition:")
    for condition_name in sorted(condition_count.keys()):
        print(f"  {condition_name}: {condition_count[condition_name]}")
    print(f"Output directory: {output_dir}")

    for save_index, record in enumerate(selected_records):
        dataset = record["dataset"]
        dataset_index = record["index"]
        sample = dataset[dataset_index]

        image_tensor = sample["image"]
        image = denormalize_image(image_tensor, config)

        input_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()

        gt_mask = None
        if "mask" in sample:
            gt_mask = sample["mask"].detach().cpu().numpy()

        sample_condition = str(sample.get("condition", record["condition"]))
        image_path = sample.get("image_path", f"sample_{dataset_index}")

        filename = f"{save_index:05d}_{sample_condition}_{safe_filename(image_path)}.png"
        output_path = output_dir / filename

        title = (
            f"source={record['source']} | "
            f"split={record['csv_split']} | "
            f"condition={sample_condition}"
        )

        save_visualization(
            image=image,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            output_path=output_path,
            config=config,
            title=title,
            alpha=alpha,
            dpi=dpi,
        )

        print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    visualize_predictions(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_index=args.start_index,
        condition=args.condition,
        samples_per_condition=args.samples_per_condition,
        include_normal=args.include_normal,
        normal_split=args.normal_split,
        shuffle=args.shuffle,
        seed=args.seed,
        alpha=args.alpha,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
