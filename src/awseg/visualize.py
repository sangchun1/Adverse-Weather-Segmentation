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
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize semantic segmentation predictions.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to visualize when condition options are not used.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index to start visualization from when not shuffling.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help=(
            "Optional condition filter, e.g. fog, rain, snow, night. "
            "Default: None, which means all conditions."
        ),
    )
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=5,
        help=(
            "If --condition is given, select N samples from that condition only. "
            "If --condition is not given, select N samples for each condition. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="Randomly shuffle selected samples. Default: true.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable random sampling and use deterministic CSV order.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


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
        panels = [
            ("Image", image),
            ("GT", gt_color),
            ("Prediction", pred_color),
            ("Overlay", pred_overlay),
        ]
    else:
        panels = [
            ("Image", image),
            ("Prediction", pred_color),
            ("Overlay", pred_overlay),
        ]

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


def select_indices(
    dataset: Any,
    num_samples: int,
    start_index: int,
    condition: Optional[str],
    samples_per_condition: Optional[int],
    shuffle: bool,
    seed: int,
) -> list[int]:
    """Select dataset indices for visualization.

    Rules:
        1. --condition + --samples-per-condition N:
           randomly select N samples from that condition only.
        2. --samples-per-condition N without --condition:
           randomly select N samples for every condition.
        3. --condition only:
           select num_samples from that condition.
        4. no condition option:
           select num_samples sequentially or randomly.
    """
    rng = random.Random(seed)

    # Case 1 and 3: one selected condition only.
    if condition is not None:
        selected = [
            idx
            for idx, sample in enumerate(dataset.samples)
            if str(sample.get("condition", "unknown")) == condition
        ]

        if shuffle:
            rng.shuffle(selected)

        n = samples_per_condition if samples_per_condition is not None else num_samples
        return selected[:n]

    # Case 2: N samples for every condition.
    if samples_per_condition is not None:
        indices_by_condition: dict[str, list[int]] = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):
            sample_condition = str(sample.get("condition", "unknown"))
            indices_by_condition[sample_condition].append(idx)

        selected: list[int] = []

        for sample_condition in sorted(indices_by_condition.keys()):
            indices = indices_by_condition[sample_condition].copy()

            if shuffle:
                rng.shuffle(indices)

            selected.extend(indices[:samples_per_condition])

        return selected

    # Case 4: default sampling.
    selected = list(range(len(dataset)))

    if shuffle:
        rng.shuffle(selected)
        return selected[:num_samples]

    end_index = min(start_index + num_samples, len(dataset))
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
    shuffle: bool,
    seed: int,
    alpha: float,
    dpi: int,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    dataset = build_dataset(config, split=split)

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)

    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    if "best_miou" in checkpoint:
        print(f"Checkpoint best mIoU: {float(checkpoint['best_miou']):.4f}")

    model.eval()

    output_dir = ensure_dir(output_dir)

    selected_indices = select_indices(
        dataset=dataset,
        num_samples=num_samples,
        start_index=start_index,
        condition=condition,
        samples_per_condition=samples_per_condition,
        shuffle=shuffle,
        seed=seed,
    )

    if len(selected_indices) == 0:
        available_conditions = sorted(
            {str(sample.get("condition", "unknown")) for sample in dataset.samples}
        )
        raise ValueError(
            f"No samples selected. condition={condition}. "
            f"Available conditions: {available_conditions}"
        )

    condition_count: dict[str, int] = defaultdict(int)
    for idx in selected_indices:
        sample_condition = str(dataset.samples[idx].get("condition", "unknown"))
        condition_count[sample_condition] += 1

    print(f"Selected {len(selected_indices)} samples")
    print("Selected samples by condition:")
    for sample_condition in sorted(condition_count.keys()):
        print(f"  {sample_condition}: {condition_count[sample_condition]}")
    print(f"Output directory: {output_dir}")

    for dataset_index in selected_indices:
        sample = dataset[dataset_index]

        image_tensor = sample["image"]
        image = denormalize_image(image_tensor, config)

        input_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()

        gt_mask = None
        if "mask" in sample:
            gt_mask = sample["mask"].detach().cpu().numpy()

        sample_condition = sample.get("condition", "unknown")
        image_path = sample.get("image_path", f"sample_{dataset_index}")

        filename = f"{dataset_index:05d}_{sample_condition}_{safe_filename(image_path)}.png"
        output_path = output_dir / filename

        title = f"index={dataset_index} | condition={sample_condition}"

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
        shuffle=args.shuffle,
        seed=args.seed,
        alpha=args.alpha,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
