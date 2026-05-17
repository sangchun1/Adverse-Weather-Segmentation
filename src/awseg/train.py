from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset

from awseg.dataset import build_dataset, get_class_names
from awseg.logger import build_logger
from awseg.losses import build_loss
from awseg.metrics import SegmentationMetric
from awseg.models import build_model
from awseg.utils import (
    AverageMeter,
    count_parameters,
    ensure_dir,
    format_metrics,
    get_device,
    get_lr,
    load_config,
    save_checkpoint,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline semantic segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help=(
            "Optional weather condition filter. "
            "Examples: fog, rain, snow, night. "
            "If set, both train and val splits use only this condition."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/baseline",
        help="Directory to save small JSON result summaries for GitHub tracking.",
    )
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        default=True,
        help="Disable saving JSON result summaries.",
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


def get_condition_suffix(condition: str | None) -> str:
    """Return filename suffix for condition-specific results.

    Examples:
        condition=None  -> ""
        condition="fog" -> "_fog"
    """
    return f"_{condition}" if condition is not None else ""


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


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    shuffle: bool,
    condition: str | None = None,
) -> DataLoader:
    dataset = build_dataset(config, split=split)

    if condition is not None:
        dataset = filter_dataset_by_condition(dataset, condition=condition, split=split)

    train_config = config["train"]
    batch_size = int(train_config["batch_size"])
    num_workers = int(train_config.get("num_workers", 4))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_optimizer(config: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Supports both config styles:

    1. Recommended style:
        optimizer:
          name: adamw
          lr: 0.001
          weight_decay: 0.0001

    2. Older flat train style:
        train:
          optimizer: adamw
          lr: 0.001
          weight_decay: 0.0001
    """
    train_config = config.get("train", {})
    optimizer_config = config.get("optimizer", {})

    optimizer_name = str(
        optimizer_config.get("name", train_config.get("optimizer", "adamw"))
    ).lower()

    lr = float(optimizer_config.get("lr", train_config.get("lr", 1e-3)))
    weight_decay = float(
        optimizer_config.get("weight_decay", train_config.get("weight_decay", 0.0))
    )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        momentum = float(optimizer_config.get("momentum", train_config.get("momentum", 0.9)))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build LR scheduler from config."""
    scheduler_config = config.get("scheduler", {})
    scheduler_name = str(scheduler_config.get("name", "cosine")).lower()

    if scheduler_name in {"none", "null", ""}:
        return None

    if scheduler_name in {"cosine", "cosine_annealing", "cosineannealinglr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("T_max", config["train"]["epochs"])),
            eta_min=float(scheduler_config.get("eta_min", 1e-6)),
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_early_stopping_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return early stopping config.

    Since this project currently asks to modify only train.py, early stopping is
    enabled by default even if configs/baseline.yaml does not contain an
    early_stopping block.

    To disable it later, add this to config:
        early_stopping:
          enabled: false
    """
    early_config = config.get("early_stopping", {})

    return {
        "enabled": bool(early_config.get("enabled", True)),
        "monitor": str(early_config.get("monitor", "val_miou")),
        "mode": str(early_config.get("mode", "max")).lower(),
        "patience": int(early_config.get("patience", 10)),
        "min_delta": float(early_config.get("min_delta", 0.0001)),
    }


def is_improved(
    current_value: float,
    best_value: float,
    mode: str,
    min_delta: float,
) -> bool:
    """Check whether monitored metric improved."""
    if mode == "max":
        return current_value > best_value + min_delta

    if mode == "min":
        return current_value < best_value - min_delta

    raise ValueError(f"Unknown early stopping mode: {mode}")


def get_initial_best_value(mode: str) -> float:
    if mode == "max":
        return float("-inf")

    if mode == "min":
        return float("inf")

    raise ValueError(f"Unknown early stopping mode: {mode}")


def load_resume_checkpoint(
    resume_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, float, int, int]:
    """Load checkpoint for resuming training.

    Returns:
        start_epoch: Epoch to start from.
        best_miou: Best validation mIoU so far.
        early_bad_epochs: Early stopping counter.
        global_step: Global iteration step for wandb logging.
    """
    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_miou = float(checkpoint.get("best_miou", 0.0))
    early_bad_epochs = int(checkpoint.get("early_bad_epochs", 0))
    global_step = int(checkpoint.get("global_step", 0))

    return start_epoch, best_miou, early_bad_epochs, global_step


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: SegmentationMetric,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: Any,
    global_step: int,
) -> tuple[Dict[str, float], int]:
    model.train()

    loss_meter = AverageMeter("train_loss")
    metric.reset()

    log_interval = int(
        config.get("train", {}).get(
            "log_interval",
            config.get("wandb", {}).get("log_interval", 20),
        )
    )

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), n=batch_size)
        metric.update(logits.detach(), masks)

        if batch_idx % log_interval == 0:
            lr = get_lr(optimizer)
            print(
                f"Epoch [{epoch}] "
                f"Train [{batch_idx:04d}/{len(dataloader):04d}] "
                f"loss: {loss_meter.avg:.4f} "
                f"lr: {lr:.6f}"
            )

            logger.log(
                {
                    "train/iter_loss": float(loss.item()),
                    "train/loss": float(loss_meter.avg),
                    "lr": float(lr),
                    "epoch": int(epoch),
                },
                step=global_step,
            )

        global_step += 1

    result = metric.compute()
    train_metrics = {
        "loss": float(loss_meter.avg),
        "miou": float(result["miou"]),
    }

    return train_metrics, global_step


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    metric: SegmentationMetric,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    loss_meter = AverageMeter("val_loss")
    metric.reset()

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), n=batch_size)
        metric.update(logits, masks)

    result = metric.compute()

    return {
        "loss": float(loss_meter.avg),
        "miou": float(result["miou"]),
        "class_iou": result["class_iou"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    save_dir = ensure_dir(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints/baseline"))
    save_config(config, save_dir / "config.yaml")

    logger = build_logger(config)

    train_loader = build_dataloader(
        config,
        split="train",
        shuffle=True,
        condition=args.condition,
    )
    val_loader = build_dataloader(
        config,
        split="val",
        shuffle=False,
        condition=args.condition,
    )

    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    metric = SegmentationMetric(
        num_classes=int(config["data"]["num_classes"]),
        ignore_index=int(config["data"].get("ignore_index", 255)),
        device=device,
    )

    class_names = get_class_names()

    print(f"Model: {config['model']['name']}")
    if args.condition is not None:
        print(f"Condition filter: {args.condition}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    logger.watch(
        model,
        log=str(config.get("wandb", {}).get("watch_log", "gradients")),
        log_freq=int(config.get("wandb", {}).get("watch_log_freq", 100)),
    )

    early_config = get_early_stopping_config(config)
    early_enabled = bool(early_config["enabled"])
    early_patience = int(early_config["patience"])
    early_min_delta = float(early_config["min_delta"])
    early_mode = str(early_config["mode"])
    early_bad_epochs = 0

    print(
        "Early stopping: "
        f"enabled={early_enabled}, "
        f"monitor={early_config['monitor']}, "
        f"mode={early_mode}, "
        f"patience={early_patience}, "
        f"min_delta={early_min_delta}"
    )

    start_epoch = 1
    best_miou = get_initial_best_value(early_mode)
    global_step = 0

    if args.resume is not None:
        start_epoch, best_miou, early_bad_epochs, global_step = load_resume_checkpoint(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resumed from {args.resume}")
        print(f"Start epoch: {start_epoch}")
        print(f"Best mIoU: {best_miou:.4f}")
        print(f"Early stopping counter: {early_bad_epochs}/{early_patience}")
        print(f"Global step: {global_step}")

    epochs = int(config["train"]["epochs"])
    training_history: list[Dict[str, Any]] = []
    stopped_early = False
    stopped_epoch = None

    result_dir = Path(args.result_dir)
    condition_suffix = get_condition_suffix(args.condition)
    condition_label = get_condition_label(args.condition)
    train_summary_path = result_dir / f"train{condition_suffix}.json"
    train_history_path = result_dir / f"train_history{condition_suffix}.json"

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_metrics, global_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                metric=metric,
                device=device,
                epoch=epoch,
                config=config,
                logger=logger,
                global_step=global_step,
            )

            val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                metric=metric,
                device=device,
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = get_lr(optimizer)
            current_miou = float(val_metrics["miou"])

            print(format_metrics(train_metrics, prefix=f"Epoch [{epoch}] Train"))
            print(
                format_metrics(
                    {
                        "loss": val_metrics["loss"],
                        "miou": current_miou,
                        "lr": current_lr,
                    },
                    prefix=f"Epoch [{epoch}] Val",
                )
            )

            # Use global_step for all wandb logs.
            # wandb step must be monotonically increasing, so do not use epoch
            # as step after iteration-level logs already used global_step.
            logger.log_metrics(train_metrics, prefix="train_epoch", step=global_step)
            logger.log_metrics(
                {
                    "loss": val_metrics["loss"],
                    "miou": current_miou,
                },
                prefix="val",
                step=global_step,
            )
            logger.log(
                {
                    "lr/epoch": float(current_lr),
                    "epoch": int(epoch),
                    "early_stopping/bad_epochs": int(early_bad_epochs),
                    "early_stopping/patience": int(early_patience),
                    "best/miou": float(best_miou) if best_miou not in {float("-inf"), float("inf")} else 0.0,
                },
                step=global_step,
            )
            logger.log_class_iou(
                val_metrics["class_iou"],
                class_names=class_names,
                prefix="val_iou",
                step=global_step,
            )

            is_best = is_improved(
                current_value=current_miou,
                best_value=best_miou,
                mode=early_mode,
                min_delta=early_min_delta,
            )

            if is_best:
                best_miou = current_miou
                early_bad_epochs = 0
                print(f"New best validation mIoU: {best_miou:.4f}")
            else:
                early_bad_epochs += 1
                print(
                    f"No improvement in validation mIoU. "
                    f"Early stopping counter: {early_bad_epochs}/{early_patience}"
                )

            checkpoint_state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_miou": best_miou,
                "early_bad_epochs": early_bad_epochs,
                "early_stopping": early_config,
                "condition": args.condition,
                "config": config,
            }

            save_checkpoint(checkpoint_state, save_dir / "last.pth")

            if is_best:
                save_checkpoint(checkpoint_state, save_dir / "best_miou.pth")

            epoch_record = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(train_metrics["loss"]),
                "train_miou": float(train_metrics["miou"]),
                "val_loss": float(val_metrics["loss"]),
                "val_miou": float(current_miou),
                "lr": float(current_lr),
                "best_miou": float(best_miou),
                "is_best": bool(is_best),
                "early_bad_epochs": int(early_bad_epochs),
            }
            training_history.append(epoch_record)

            if args.save_results:
                save_json(training_history, train_history_path)

            if early_enabled and early_bad_epochs >= early_patience:
                stopped_early = True
                stopped_epoch = epoch
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best validation mIoU: {best_miou:.4f}"
                )
                logger.log(
                    {
                        "early_stopping/triggered": 1,
                        "early_stopping/stopped_epoch": int(epoch),
                        "best/miou": float(best_miou),
                    },
                    step=global_step,
                )
                break

        print(f"Training finished. Best validation mIoU: {best_miou:.4f}")

        if args.save_results:
            train_summary = {
                "task": "train",
                "created_at": get_timestamp(),
                "config_path": str(args.config),
                "condition": condition_label,
                "model": config.get("model", {}),
                "optimizer": config.get("optimizer", {}),
                "scheduler": config.get("scheduler", {}),
                "loss": config.get("loss", {}),
                "input_size": {
                    "height": int(config["data"]["input_height"]),
                    "width": int(config["data"]["input_width"]),
                },
                "num_classes": int(config["data"]["num_classes"]),
                "ignore_index": int(config["data"].get("ignore_index", 255)),
                "epochs_requested": int(epochs),
                "epochs_completed": int(len(training_history)),
                "best_miou": float(best_miou),
                "best_epoch": (
                    max(training_history, key=lambda x: x["val_miou"])["epoch"]
                    if len(training_history) > 0
                    else None
                ),
                "final_train_loss": (
                    float(training_history[-1]["train_loss"])
                    if len(training_history) > 0
                    else None
                ),
                "final_train_miou": (
                    float(training_history[-1]["train_miou"])
                    if len(training_history) > 0
                    else None
                ),
                "final_val_loss": (
                    float(training_history[-1]["val_loss"])
                    if len(training_history) > 0
                    else None
                ),
                "final_val_miou": (
                    float(training_history[-1]["val_miou"])
                    if len(training_history) > 0
                    else None
                ),
                "stopped_early": bool(stopped_early),
                "stopped_epoch": stopped_epoch,
                "early_stopping": early_config,
                "num_train_samples": int(len(train_loader.dataset)),
                "num_val_samples": int(len(val_loader.dataset)),
                "best_checkpoint": str(save_dir / "best_miou.pth"),
                "last_checkpoint": str(save_dir / "last.pth"),
                "history_file": str(train_history_path),
                "history": training_history,
            }
            save_json(train_summary, train_summary_path)
            print(f"Saved train result JSON: {train_summary_path}")
            print(f"Saved train history JSON: {train_history_path}")

        print("Best checkpoint:", save_dir / "best_miou.pth")

        if (save_dir / "best_miou.pth").exists():
            print("\nFinal best checkpoint can be evaluated with:")
            condition_arg = f" --condition {args.condition}" if args.condition is not None else ""
            print(
                f"python -m awseg.evaluate --config {args.config} "
                f"--checkpoint {save_dir / 'best_miou.pth'} --split val{condition_arg}"
            )

    finally:
        logger.finish()


if __name__ == "__main__":
    main()
