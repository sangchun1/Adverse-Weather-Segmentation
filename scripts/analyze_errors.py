#!/usr/bin/env python
"""General segmentation error analysis for Adverse-Weather-Segmentation.

This script is intended for the final main branch. It works across experiment
families such as baseline, loss, model, augmentation, enhancement, and proposed.

Main outputs:
- per-image error summary CSV
- confusion matrix CSV
- class error summary CSV
- top confusion pairs CSV/plot
- boundary error analysis
- dynamic-object -> background confusion analysis
- representative example visualizations with error maps
- optional class IoU drop CSV/plots when condition-specific eval JSONs exist

Examples:
    # Whole validation-set analysis
    python scripts/analyze_errors.py \
        --group baseline \
        --condition none \
        --device cuda:1

    # Condition-specific analysis for an enhancement experiment
    python scripts/analyze_errors.py \
        --group enhancement \
        --experiment gamma_clahe \
        --condition night \
        --device cuda:1

    # Explicit paths always override inferred paths
    python scripts/analyze_errors.py \
        --config configs/proposed.yaml \
        --checkpoint outputs/checkpoints/proposed/best_miou.pth \
        --condition fog \
        --output-dir outputs/analysis/proposed/fog
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from awseg.dataset import build_dataset  # noqa: E402
from awseg.models import build_model  # noqa: E402

try:  # noqa: E402
    from awseg.dataset import get_class_names as _get_class_names
except ImportError:  # pragma: no cover
    _get_class_names = None

try:  # noqa: E402
    from awseg.utils import get_device as _get_device
except ImportError:  # pragma: no cover
    _get_device = None

try:  # noqa: E402
    from awseg.utils import load_config as _load_config
except ImportError:  # pragma: no cover
    _load_config = None


ACDC_CLASS_NAMES = [
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

CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
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

ERROR_COLORS = {
    "other_error": np.array([0, 128, 255], dtype=np.uint8),
    "boundary_error": np.array([255, 0, 0], dtype=np.uint8),
    "dynamic_to_background": np.array([255, 255, 0], dtype=np.uint8),
}

ALL_CONDITION_VALUES = {"", "none", "all", "overall", "val", "baseline"}
SUPPORTED_GROUPS = {"baseline", "loss", "model", "augmentation", "enhancement", "proposed"}


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


# -----------------------------------------------------------------------------
# Argument and path handling
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze semantic segmentation errors for a whole split or one condition."
    )
    parser.add_argument("--config", type=str, default=None, help="Config path. Overrides --group/--experiment inference.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Overrides inferred checkpoint path.")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_GROUPS),
        help="Experiment group used for automatic path inference.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name inside the group, e.g. ce_dice, segformer, gamma_clahe.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--condition",
        type=str,
        default="none",
        help="none/all for whole split, or fog/night/rain/snow for condition-specific analysis.",
    )
    parser.add_argument("--result-root", type=str, default="outputs/results")
    parser.add_argument("--checkpoint-root", type=str, default="outputs/checkpoints")
    parser.add_argument("--analysis-root", type=str, default="outputs/analysis")
    parser.add_argument(
        "--overall-result",
        type=str,
        default=None,
        help="Whole-split evaluation JSON. Used for class_iou_drop.csv when condition != none.",
    )
    parser.add_argument(
        "--condition-result",
        "--night-result",
        dest="condition_result",
        type=str,
        default=None,
        help="Condition-specific evaluation JSON. --night-result is kept for backward compatibility.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Overrides inferred path.")
    parser.add_argument("--top-k", type=int, default=5, help="Representative examples per category.")
    parser.add_argument("--boundary-radius", type=int, default=2, help="GT boundary band radius in pixels.")
    parser.add_argument(
        "--min-dynamic-pixels",
        type=int,
        default=50,
        help="Minimum GT dynamic pixels for dynamic-to-background examples.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Visualization overlay alpha.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--device", type=str, default=None, help="cuda:0, cuda:1, cpu, etc.")
    return parser.parse_args()


def is_all_condition(condition: Optional[str]) -> bool:
    if condition is None:
        return True
    return str(condition).strip().lower() in ALL_CONDITION_VALUES


def sanitize_name(value: str) -> str:
    value = str(value).strip().lower()
    return re.sub(r"[^a-zA-Z0-9_.=-]+", "_", value).strip("_")


def get_condition_name(condition: Optional[str]) -> str:
    if is_all_condition(condition):
        return "all"
    return sanitize_name(str(condition)) or "condition"


def get_experiment_key(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    group = args.group
    experiment = args.experiment

    if group is None:
        if args.config:
            config_path = Path(args.config)
            parts = config_path.parts
            if "configs" in parts:
                idx = parts.index("configs")
                if len(parts) > idx + 2:
                    group = parts[idx + 1]
                    experiment = experiment or config_path.stem
                else:
                    stem = config_path.stem
                    group = "baseline" if stem == "baseline" else "proposed" if stem == "proposed" else stem
            else:
                group = "custom"
        else:
            group = "baseline"

    if group in {"baseline", "proposed"}:
        experiment = None
    elif not experiment:
        raise ValueError(f"--experiment is required when --group {group!r} is used.")

    return group, experiment


def infer_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)

    group, experiment = get_experiment_key(args)
    if group == "baseline":
        return Path("configs/baseline.yaml")
    if group == "proposed":
        return Path("configs/proposed.yaml")
    return Path("configs") / group / f"{experiment}.yaml"


def infer_run_dir(root: str | Path, group: str, experiment: Optional[str]) -> Path:
    root = Path(root)
    if group in {"baseline", "proposed"}:
        return root / group
    return root / group / str(experiment)


def infer_checkpoint_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)

    checkpoint_config = config.get("checkpoint", {}) if isinstance(config.get("checkpoint", {}), dict) else {}
    save_dir = checkpoint_config.get("save_dir")
    save_best_name = checkpoint_config.get("save_best_name", "best_miou.pth")
    if save_dir:
        return Path(save_dir) / str(save_best_name)

    group, experiment = get_experiment_key(args)
    return infer_run_dir(args.checkpoint_root, group, experiment) / "best_miou.pth"


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    group, experiment = get_experiment_key(args)
    base = infer_run_dir(args.analysis_root, group, experiment)
    if is_all_condition(args.condition):
        return base
    return base / get_condition_name(args.condition)


def infer_result_json(args: argparse.Namespace, condition_specific: bool) -> Path:
    group, experiment = get_experiment_key(args)
    result_dir = infer_run_dir(args.result_root, group, experiment)
    if condition_specific and not is_all_condition(args.condition):
        return result_dir / f"eval_{args.split}_{get_condition_name(args.condition)}.json"
    return result_dir / f"eval_{args.split}.json"


# -----------------------------------------------------------------------------
# Basic IO and config utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    if _load_config is not None:
        return _load_config(str(path))
    if yaml is None:
        raise ImportError("PyYAML is required when awseg.utils.load_config is unavailable.")
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if _get_device is not None:
        return _get_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names() -> list[str]:
    if _get_class_names is not None:
        return list(_get_class_names())
    return ACDC_CLASS_NAMES.copy()


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


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(value: str, max_len: int = 120) -> str:
    name = Path(str(value)).stem or "sample"
    name = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", name).strip("_")
    return (name or "sample")[:max_len]


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else float("nan")


def format_float(value: float) -> str:
    return "nan" if math.isnan(float(value)) else f"{float(value):.4f}"


# -----------------------------------------------------------------------------
# Class and metric helpers
# -----------------------------------------------------------------------------


def sanitize_class_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("-", "_")


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
        if name in mapping:
            ids.append(mapping[name])
        elif key in mapping:
            ids.append(mapping[key])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Unknown class names: {missing}. Available classes: {class_names}")
    return ids


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


def make_semantic_boundary_band(gt: np.ndarray, valid_mask: np.ndarray, radius: int) -> np.ndarray:
    boundary = np.zeros_like(valid_mask, dtype=bool)
    vertical_diff = valid_mask[:-1, :] & valid_mask[1:, :] & (gt[:-1, :] != gt[1:, :])
    boundary[:-1, :] |= vertical_diff
    boundary[1:, :] |= vertical_diff
    horizontal_diff = valid_mask[:, :-1] & valid_mask[:, 1:] & (gt[:, :-1] != gt[:, 1:])
    boundary[:, :-1] |= horizontal_diff
    boundary[:, 1:] |= horizontal_diff

    if radius <= 0:
        return boundary & valid_mask

    kernel_size = 2 * radius + 1
    boundary_tensor = torch.from_numpy(boundary.astype(np.float32))[None, None]
    dilated = F.max_pool2d(boundary_tensor, kernel_size=kernel_size, stride=1, padding=radius)[0, 0]
    return (dilated.numpy() > 0) & valid_mask


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
            raise ValueError(f"class_iou length mismatch: {len(class_iou)} vs {len(class_names)}")
        return {
            class_name: float(value) if value is not None else float("nan")
            for class_name, value in zip(class_names, class_iou)
        }

    raise TypeError(f"Unsupported class_iou type: {type(class_iou)}")


# -----------------------------------------------------------------------------
# Model inference helpers
# -----------------------------------------------------------------------------


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    state_dict = _strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Warning: loaded checkpoint with strict=False")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state_dict}


def get_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict):
        if "logits" in model_output:
            return model_output["logits"]
        if "out" in model_output:
            return model_output["out"]
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (list, tuple)) and model_output:
        return get_logits(model_output[0])
    raise TypeError(f"Cannot extract logits from model output type: {type(model_output)}")


def predict_mask(model: torch.nn.Module, image_tensor: torch.Tensor, gt_shape: tuple[int, int], device: torch.device) -> np.ndarray:
    logits = get_logits(model(image_tensor.unsqueeze(0).to(device, non_blocking=True)))
    if logits.shape[-2:] != gt_shape:
        logits = F.interpolate(logits, size=gt_shape, mode="bilinear", align_corners=False)
    pred = torch.argmax(logits, dim=1)[0]
    return pred.detach().cpu().numpy().astype(np.int64)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def colorize_mask(mask: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    palette = CITYSCAPES_PALETTE
    for class_id in range(num_classes):
        color = palette[class_id % len(palette)]
        colored[mask == class_id] = color
    colored[mask == ignore_index] = np.array([0, 0, 0], dtype=np.uint8)
    return colored


def _get_mean_std(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        config.get("data", {}),
        config.get("transform", {}),
        config.get("transforms", {}),
        config.get("dataset", {}),
    ]
    mean = None
    std = None
    for block in candidates:
        if isinstance(block, dict):
            mean = mean or block.get("mean") or block.get("image_mean")
            std = std or block.get("std") or block.get("image_std")
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)


def denormalize_image(image_tensor: torch.Tensor, config: dict[str, Any]) -> np.ndarray:
    image = image_tensor.detach().cpu().float().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {image.shape}")
    image = np.transpose(image, (1, 2, 0))
    mean, std = _get_mean_std(config)
    image = image * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def make_overlay(image: np.ndarray, mask_color: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * image.astype(np.float32) + alpha * mask_color.astype(np.float32), 0, 255).astype(
        np.uint8
    )


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
    error_map[wrong] = ERROR_COLORS["other_error"]
    error_map[wrong & boundary_band] = ERROR_COLORS["boundary_error"]
    error_map[wrong & dynamic_to_background] = ERROR_COLORS["dynamic_to_background"]
    return error_map


def make_error_overlay(image: np.ndarray, error_map: np.ndarray, alpha: float) -> np.ndarray:
    has_error = error_map.sum(axis=2) > 0
    overlay = image.astype(np.float32).copy()
    overlay[has_error] = (1.0 - alpha) * overlay[has_error] + alpha * error_map[has_error].astype(np.float32)
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
    num_classes = int(config.get("data", {}).get("num_classes", len(ACDC_CLASS_NAMES)))
    ignore_index = int(config.get("data", {}).get("ignore_index", 255))
    gt_color = colorize_mask(gt, num_classes=num_classes, ignore_index=ignore_index)
    pred_color = colorize_mask(pred, num_classes=num_classes, ignore_index=ignore_index)
    panels = [
        ("Image", image),
        ("GT", gt_color),
        ("Prediction", pred_color),
        ("Error map\nred=boundary, yellow=dynamic→bg, blue=other", error_map),
        ("Error overlay", make_error_overlay(image, error_map, alpha=alpha)),
        ("Pred overlay", make_overlay(image, pred_color, alpha=alpha)),
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


# -----------------------------------------------------------------------------
# Dataset/sample helpers
# -----------------------------------------------------------------------------


def get_sample_meta(dataset: Any, index: int) -> dict[str, Any]:
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and index < len(samples) and isinstance(samples[index], dict):
        return samples[index]
    return {}


def get_sample_condition(meta: dict[str, Any], sample: Optional[dict[str, Any]] = None) -> str:
    if sample is not None and "condition" in sample:
        return str(sample.get("condition", "unknown"))
    return str(meta.get("condition", "unknown"))


def select_indices(dataset: Any, condition: str, max_samples: Optional[int]) -> list[int]:
    if is_all_condition(condition):
        indices = list(range(len(dataset)))
    elif hasattr(dataset, "samples"):
        target = str(condition).strip().lower()
        indices = [
            idx
            for idx in range(len(dataset))
            if str(get_sample_meta(dataset, idx).get("condition", "unknown")).strip().lower() == target
        ]
    else:
        target = str(condition).strip().lower()
        indices = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_condition = str(sample.get("condition", "unknown")).strip().lower()
            if sample_condition == target:
                indices.append(idx)

    if max_samples is not None:
        indices = indices[:max_samples]
    return indices


# -----------------------------------------------------------------------------
# Analysis outputs
# -----------------------------------------------------------------------------


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

    rows = []
    for class_id, class_name in enumerate(class_names):
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
                "class_id": class_id,
                "class_name": class_name,
                "overall_val_iou": overall_value,
                "condition_iou": condition_value,
                "iou_drop_overall_minus_condition": drop,
                "condition_minus_overall": condition_minus_overall,
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: -1e9 if math.isnan(float(r["iou_drop_overall_minus_condition"])) else float(r["iou_drop_overall_minus_condition"]),
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
    names = [str(row["class_name"]) for row in rows]
    overall = np.array([float(row["overall_val_iou"]) for row in rows], dtype=np.float64)
    condition = np.array([float(row["condition_iou"]) for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.55), 5))
    ax.bar(x - width / 2, overall, width, label="overall val")
    ax.bar(x + width / 2, condition, width, label="condition val")
    ax.set_ylabel("IoU")
    ax.set_title("Class-wise IoU: overall val vs condition")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_iou_drop(rows_sorted: list[dict[str, Any]], output_path: Path) -> None:
    valid_rows = [row for row in rows_sorted if not math.isnan(float(row["iou_drop_overall_minus_condition"]))]
    if not valid_rows:
        return
    names = [str(row["class_name"]) for row in valid_rows]
    drops = np.array([float(row["iou_drop_overall_minus_condition"]) for row in valid_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(valid_rows))))
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


def analyze_predictions(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    condition: str,
    output_dir: Path,
    boundary_radius: int,
    max_samples: Optional[int],
    device: torch.device,
) -> tuple[list[ImageErrorRecord], np.ndarray, dict[str, np.ndarray]]:
    dataset = build_dataset(config, split=split)
    class_names = get_class_names()
    num_classes = int(config.get("data", {}).get("num_classes", len(class_names)))
    ignore_index = int(config.get("data", {}).get("ignore_index", 255))
    selected_indices = select_indices(dataset, condition=condition, max_samples=max_samples)

    if not selected_indices:
        if hasattr(dataset, "samples"):
            available = sorted({str(sample.get("condition", "unknown")) for sample in dataset.samples})
        else:
            available = ["unknown"]
        raise ValueError(f"No samples found for condition={condition!r}. Available conditions: {available}")

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selected samples ({condition}): {len(selected_indices)} / {len(dataset)}")

    model = build_model(config).to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    if isinstance(checkpoint, dict):
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
            meta = get_sample_meta(dataset, dataset_index)
            sample = dataset[dataset_index]
            if "mask" not in sample:
                raise ValueError(f"Sample does not contain mask. split={split}, index={dataset_index}")

            image_tensor = sample["image"]
            gt_tensor = sample["mask"]
            gt = gt_tensor.detach().cpu().numpy().astype(np.int64)
            pred = predict_mask(model, image_tensor, gt_shape=gt.shape, device=device)

            valid = gt != ignore_index
            valid &= gt >= 0
            valid &= gt < num_classes
            valid &= pred >= 0
            valid &= pred < num_classes
            wrong = (pred != gt) & valid

            conf = compute_confusion_matrix_np(pred, gt, num_classes, ignore_index)
            confusion += conf
            image_miou, _ = compute_iou_from_confusion(conf)

            boundary_band = make_semantic_boundary_band(gt=gt, valid_mask=valid, radius=boundary_radius)
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
            dynamic_to_background_error_pixels = int((wrong & dynamic_to_background).sum())

            records.append(
                ImageErrorRecord(
                    dataset_index=dataset_index,
                    image_path=str(sample.get("image_path", meta.get("image_path", ""))),
                    label_path=str(sample.get("label_path", meta.get("label_path", ""))),
                    condition=get_sample_condition(meta, sample),
                    image_miou=image_miou,
                    pixel_error_rate=safe_ratio(error_pixels, valid_pixels),
                    boundary_error_rate=safe_ratio(boundary_error_pixels, boundary_pixels),
                    interior_error_rate=safe_ratio(interior_error_pixels, interior_pixels),
                    dynamic_to_background_error_rate=safe_ratio(dynamic_to_background_error_pixels, gt_dynamic_pixels),
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


def save_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    rows = []
    for gt_id, gt_name in enumerate(class_names):
        row = {"gt_class_id": gt_id, "gt_class_name": gt_name}
        for pred_id, pred_name in enumerate(class_names):
            row[f"pred_{pred_id}_{sanitize_class_name(pred_name)}"] = int(confusion[gt_id, pred_id])
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "confusion_matrix.csv", rows, fieldnames)


def make_class_error_summary(
    confusion: np.ndarray,
    class_names: list[str],
    boundary_pixels_by_class: np.ndarray,
    boundary_errors_by_class: np.ndarray,
    output_dir: Path,
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

    rows = []
    for class_id, class_name in enumerate(class_names):
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": int(gt_pixels[class_id]),
                "pred_pixels": int(pred_pixels[class_id]),
                "true_positive": int(tp[class_id]),
                "false_positive": int(fp[class_id]),
                "false_negative": int(fn[class_id]),
                "iou": safe_ratio(tp[class_id], union[class_id]),
                "recall": safe_ratio(tp[class_id], gt_pixels[class_id]),
                "precision": safe_ratio(tp[class_id], pred_pixels[class_id]),
                "class_error_rate_gt_missed": safe_ratio(fn[class_id], gt_pixels[class_id]),
                "boundary_pixels": int(boundary_pixels_by_class[class_id]),
                "boundary_error_pixels": int(boundary_errors_by_class[class_id]),
                "boundary_error_rate": safe_ratio(boundary_errors_by_class[class_id], boundary_pixels_by_class[class_id]),
                "dynamic_to_background_rate": safe_ratio(confusion[class_id, background_ids].sum(), gt_pixels[class_id])
                if class_id in dynamic_ids
                else float("nan"),
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: -1.0 if math.isnan(float(r["class_error_rate_gt_missed"])) else float(r["class_error_rate_gt_missed"]),
        reverse=True,
    )
    write_csv(
        output_dir / "class_error_summary.csv",
        rows_sorted,
        fieldnames=list(rows_sorted[0].keys()) if rows_sorted else [],
    )
    return rows_sorted


def make_top_confusion_pairs(
    confusion: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    top_n: int = 30,
) -> list[dict[str, Any]]:
    rows = []
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
    rows_sorted = sorted(rows, key=lambda row: int(row["pixels"]), reverse=True)
    write_csv(
        output_dir / "top_confusion_pairs.csv",
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


def plot_top_confusion_pairs(rows: list[dict[str, Any]], output_path: Path, top_n: int = 15) -> None:
    rows = rows[:top_n]
    if not rows:
        return
    labels = [f"{row['gt_class_name']} → {row['pred_class_name']}" for row in rows]
    values = np.array([float(row["pixels"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified pixels")
    ax.set_title("Top confusion pairs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_error_rates(records: list[ImageErrorRecord], output_path: Path) -> None:
    if not records:
        return
    sorted_records = sorted(records, key=lambda record: record.pixel_error_rate, reverse=True)
    x = np.arange(len(sorted_records))
    pixel_error = np.array([record.pixel_error_rate for record in sorted_records], dtype=np.float64)
    boundary_error = np.array([record.boundary_error_rate for record in sorted_records], dtype=np.float64)
    interior_error = np.array([record.interior_error_rate for record in sorted_records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, pixel_error, label="overall pixel error")
    ax.plot(x, boundary_error, label="boundary error")
    ax.plot(x, interior_error, label="interior error")
    ax.set_xlabel("Images sorted by overall pixel error")
    ax.set_ylabel("Error rate")
    ax.set_title("Per-image error rates")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_background_rates(records: list[ImageErrorRecord], output_path: Path) -> None:
    valid_records = [record for record in records if not math.isnan(record.dynamic_to_background_error_rate)]
    if not valid_records:
        return
    sorted_records = sorted(valid_records, key=lambda record: record.dynamic_to_background_error_rate, reverse=True)
    rates = np.array([record.dynamic_to_background_error_rate for record in sorted_records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(rates)), rates)
    ax.set_xlabel("Images with dynamic GT pixels, sorted by rate")
    ax.set_ylabel("Dynamic → background rate")
    ax.set_title("Dynamic-object to background confusion per image")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_selected_examples(
    records_by_category: dict[str, list[ImageErrorRecord]],
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
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
    num_classes = int(config.get("data", {}).get("num_classes", len(class_names)))
    ignore_index = int(config.get("data", {}).get("ignore_index", 255))
    dynamic_ids = ids_from_names(DYNAMIC_CLASSES, class_names)
    background_ids = ids_from_names(BACKGROUND_CLASSES, class_names)
    manifest = []

    with torch.no_grad():
        for category, records in records_by_category.items():
            category_dir = ensure_dir(output_dir / "examples" / category)
            for rank, record in enumerate(records, start=1):
                sample = dataset[record.dataset_index]
                image_tensor = sample["image"]
                gt = sample["mask"].detach().cpu().numpy().astype(np.int64)
                pred = predict_mask(model, image_tensor, gt_shape=gt.shape, device=device)
                valid = gt != ignore_index
                valid &= gt >= 0
                valid &= gt < num_classes
                valid &= pred >= 0
                valid &= pred < num_classes
                boundary_band = make_semantic_boundary_band(gt=gt, valid_mask=valid, radius=boundary_radius)
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
                image = denormalize_image(image_tensor, config)
                image_path = record.image_path or str(sample.get("image_path", f"sample_{record.dataset_index}"))
                filename = (
                    f"{rank:02d}_idx{record.dataset_index:05d}_"
                    f"miou{format_float(record.image_miou)}_"
                    f"err{format_float(record.pixel_error_rate)}_"
                    f"{safe_filename(image_path)}.png"
                )
                out_path = category_dir / filename
                save_error_visualization(
                    image=image,
                    gt=gt,
                    pred=pred,
                    error_map=error_map,
                    output_path=out_path,
                    config=config,
                    title=f"{category} | idx={record.dataset_index} | condition={record.condition}",
                    alpha=alpha,
                    dpi=dpi,
                )
                manifest.append(
                    {
                        "category": category,
                        "rank": rank,
                        "dataset_index": record.dataset_index,
                        "condition": record.condition,
                        "image_path": image_path,
                        "visualization_path": str(out_path),
                        "image_miou": record.image_miou,
                        "pixel_error_rate": record.pixel_error_rate,
                        "boundary_error_rate": record.boundary_error_rate,
                        "dynamic_to_background_error_rate": record.dynamic_to_background_error_rate,
                    }
                )

    write_csv(
        output_dir / "examples_manifest.csv",
        manifest,
        fieldnames=list(manifest[0].keys()) if manifest else [],
    )


def select_representative_records(
    records: list[ImageErrorRecord],
    top_k: int,
    min_dynamic_pixels: int,
) -> dict[str, list[ImageErrorRecord]]:
    worst_miou = sorted(records, key=lambda record: record.image_miou)[:top_k]
    best_miou = sorted(records, key=lambda record: record.image_miou, reverse=True)[:top_k]
    highest_pixel_error = sorted(records, key=lambda record: record.pixel_error_rate, reverse=True)[:top_k]
    highest_boundary_error = sorted(
        [record for record in records if not math.isnan(record.boundary_error_rate)],
        key=lambda record: record.boundary_error_rate,
        reverse=True,
    )[:top_k]
    highest_dynamic_to_background = sorted(
        [
            record
            for record in records
            if record.gt_dynamic_pixels >= min_dynamic_pixels and not math.isnan(record.dynamic_to_background_error_rate)
        ],
        key=lambda record: record.dynamic_to_background_error_rate,
        reverse=True,
    )[:top_k]
    return {
        "worst_miou": worst_miou,
        "best_miou": best_miou,
        "highest_pixel_error": highest_pixel_error,
        "highest_boundary_error": highest_boundary_error,
        "highest_dynamic_to_background": highest_dynamic_to_background,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config_path = infer_config_path(args)
    config = load_config(config_path)
    checkpoint_path = infer_checkpoint_path(args, config)
    output_dir = ensure_dir(infer_output_dir(args))
    plot_dir = ensure_dir(output_dir / "plots")
    device = get_device(args.device)

    class_names = get_class_names()
    group, experiment = get_experiment_key(args)

    print(f"Config: {config_path}")
    print(f"Group: {group}")
    print(f"Experiment: {experiment or group}")
    print(f"Condition: {args.condition}")
    print(f"Output directory: {output_dir}")

    # Optional class-IoU drop analysis from already-computed eval JSONs.
    if not is_all_condition(args.condition):
        overall_result = Path(args.overall_result) if args.overall_result else infer_result_json(args, condition_specific=False)
        condition_result = Path(args.condition_result) if args.condition_result else infer_result_json(args, condition_specific=True)
        if overall_result.exists() and condition_result.exists():
            make_class_iou_drop_outputs(overall_result, condition_result, class_names, output_dir)
        else:
            print("Skipping class_iou_drop.csv because result JSONs were not found:")
            print(f"  overall_result:   {overall_result}")
            print(f"  condition_result: {condition_result}")

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

    record_rows = [asdict(record) for record in records]
    write_csv(
        output_dir / "per_image_errors.csv",
        record_rows,
        fieldnames=list(record_rows[0].keys()) if record_rows else [],
    )

    save_confusion_matrix(confusion, class_names, output_dir)
    class_rows = make_class_error_summary(
        confusion=confusion,
        class_names=class_names,
        boundary_pixels_by_class=extras["boundary_pixels_by_class"],
        boundary_errors_by_class=extras["boundary_errors_by_class"],
        output_dir=output_dir,
    )
    top_confusions = make_top_confusion_pairs(confusion, class_names, output_dir)

    plot_top_confusion_pairs(top_confusions, plot_dir / "top_confusion_pairs.png")
    plot_image_error_rates(records, plot_dir / "per_image_error_rates.png")
    plot_dynamic_background_rates(records, plot_dir / "dynamic_to_background_rates.png")

    selected = select_representative_records(records, top_k=args.top_k, min_dynamic_pixels=args.min_dynamic_pixels)
    save_selected_examples(
        records_by_category=selected,
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        output_dir=output_dir,
        boundary_radius=args.boundary_radius,
        alpha=args.alpha,
        dpi=args.dpi,
        device=device,
    )

    miou, class_iou = compute_iou_from_confusion(confusion)
    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "group": group,
        "experiment": experiment or group,
        "split": args.split,
        "condition": args.condition,
        "num_images": len(records),
        "miou_from_selected_samples": miou,
        "mean_pixel_error_rate": float(np.nanmean([record.pixel_error_rate for record in records])) if records else float("nan"),
        "mean_boundary_error_rate": float(np.nanmean([record.boundary_error_rate for record in records])) if records else float("nan"),
        "mean_dynamic_to_background_error_rate": float(
            np.nanmean([record.dynamic_to_background_error_rate for record in records])
        )
        if records
        else float("nan"),
        "class_iou_from_selected_samples": {
            class_name: float(value) if not math.isnan(float(value)) else None
            for class_name, value in zip(class_names, class_iou)
        },
        "outputs": {
            "per_image_errors": str(output_dir / "per_image_errors.csv"),
            "class_error_summary": str(output_dir / "class_error_summary.csv"),
            "top_confusion_pairs": str(output_dir / "top_confusion_pairs.csv"),
            "confusion_matrix": str(output_dir / "confusion_matrix.csv"),
            "plots": str(plot_dir),
            "examples": str(output_dir / "examples"),
        },
    }
    save_json(summary, output_dir / "summary.json")

    print("Done.")
    print(f"Selected-sample mIoU: {miou:.4f}")
    print(f"Saved analysis to: {output_dir}")
    if class_rows:
        print("Top classes by GT-missed error rate:")
        for row in class_rows[:5]:
            print(f"  {row['class_name']}: {format_float(row['class_error_rate_gt_missed'])}")


if __name__ == "__main__":
    main()
