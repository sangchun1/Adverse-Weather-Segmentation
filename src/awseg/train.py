from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from awseg.dataset import build_dataset, get_class_names
from awseg.logger import build_logger
from awseg.losses import build_loss
from awseg.metrics import SegmentationMetric, format_class_iou
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
    return parser.parse_args()


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    shuffle: bool,
) -> DataLoader:
    dataset = build_dataset(config, split=split)

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
    train_config = config["train"]
    optimizer_name = str(train_config.get("optimizer", "adamw")).lower()

    lr = float(train_config["lr"])
    weight_decay = float(train_config.get("weight_decay", 0.0))

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
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_config = config.get("scheduler", {})
    scheduler_name = str(scheduler_config.get("name", "cosine")).lower()

    if scheduler_name in {"none", "null", ""}:
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("T_max", config["train"]["epochs"])),
            eta_min=float(scheduler_config.get("eta_min", 1e-6)),
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def load_resume_checkpoint(
    resume_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_miou = float(checkpoint.get("best_miou", 0.0))

    return start_epoch, best_miou


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

    log_interval = int(config.get("wandb", {}).get("log_interval", 20))

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

    save_dir = ensure_dir(config.get("checkpoint", {}).get("save_dir", "checkpoints/baseline"))
    save_config(config, save_dir / "config.yaml")

    logger = build_logger(config)

    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)

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
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    logger.watch(
        model,
        log=str(config.get("wandb", {}).get("watch_log", "gradients")),
        log_freq=int(config.get("wandb", {}).get("watch_log_freq", 100)),
    )

    start_epoch = 1
    best_miou = 0.0

    if args.resume is not None:
        start_epoch, best_miou = load_resume_checkpoint(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resumed from {args.resume}")
        print(f"Start epoch: {start_epoch}, best mIoU: {best_miou:.4f}")

    epochs = int(config["train"]["epochs"])
    global_step = (start_epoch - 1) * len(train_loader)

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

            print(format_metrics(train_metrics, prefix=f"Epoch [{epoch}] Train"))
            print(
                format_metrics(
                    {
                        "loss": val_metrics["loss"],
                        "miou": val_metrics["miou"],
                        "lr": current_lr,
                    },
                    prefix=f"Epoch [{epoch}] Val",
                )
            )

            logger.log_metrics(train_metrics, prefix="train_epoch", step=epoch)
            logger.log_metrics(
                {
                    "loss": val_metrics["loss"],
                    "miou": val_metrics["miou"],
                },
                prefix="val",
                step=epoch,
            )
            logger.log({"lr/epoch": float(current_lr), "epoch": int(epoch)}, step=epoch)
            logger.log_class_iou(
                val_metrics["class_iou"],
                class_names=class_names,
                prefix="val_iou",
                step=epoch,
            )

            is_best = val_metrics["miou"] > best_miou

            if is_best:
                best_miou = float(val_metrics["miou"])
                print(f"New best validation mIoU: {best_miou:.4f}")

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_miou": best_miou,
                "config": config,
            }

            save_checkpoint(checkpoint_state, save_dir / "last.pth")

            if is_best:
                save_checkpoint(checkpoint_state, save_dir / "best_miou.pth")

        print(f"Training finished. Best validation mIoU: {best_miou:.4f}")
        print("Best checkpoint:", save_dir / "best_miou.pth")

        if (save_dir / "best_miou.pth").exists():
            print("\nFinal best checkpoint can be evaluated with:")
            print(
                f"python -m awseg.evaluate --config {args.config} "
                f"--checkpoint {save_dir / 'best_miou.pth'} --split val"
            )

    finally:
        logger.finish()


if __name__ == "__main__":
    main()
