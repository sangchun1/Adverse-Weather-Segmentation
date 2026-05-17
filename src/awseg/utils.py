from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    """Save config as YAML.

    Args:
        config: Config dictionary.
        save_path: Output YAML path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def save_config_as_json(config: Dict[str, Any], save_path: str | Path) -> None:
    """Save config as JSON.

    This is useful for logging experiment settings next to checkpoints.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, use deterministic CuDNN behavior.
            This may make training slower.
    """
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return available training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object of the created directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AverageMeter:
    """Track running average of a scalar value."""

    def __init__(self, name: str = "meter") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: val={self.val:.4f}, avg={self.avg:.4f}"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only parameters with requires_grad=True.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    state: Dict[str, Any],
    save_path: str | Path,
) -> None:
    """Save a training checkpoint.

    Expected state example:
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": config,
        }

    Args:
        state: Checkpoint state dictionary.
        save_path: Output checkpoint path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        map_location: Device mapping for torch.load.

    Returns:
        Checkpoint dictionary.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if map_location is None:
        map_location = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metric dictionary for console logging.

    Args:
        metrics: Dictionary of metric name to value.
        prefix: Optional text prefix.

    Returns:
        Formatted string.
    """
    parts = []

    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")

    text = " | ".join(parts)

    if prefix:
        return f"{prefix} | {text}"

    return text
