from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from awseg.dataset import build_dataset
from awseg.losses import build_loss
from awseg.metrics import SegmentationMetric
from awseg.models import build_model
from awseg.models.frequency import dst_channel, flatten_spatial, idst_channel, reject_frequency_band, unflatten_spatial
from awseg.utils import ensure_dir, get_device, load_config, set_seed


class BinaryConditionDataset(Dataset):
    """Wrap a segmentation dataset and add a binary condition label."""

    def __init__(self, dataset: Dataset, label: int) -> None:
        self.dataset = dataset
        self.label = int(label)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["binary_label"] = torch.tensor(self.label, dtype=torch.long)
        return item


class BandMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = min(int(hidden_dim), max(32, int(in_dim)))
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NightAdapter-style band rejection analysis for SegFormer features."
    )
    parser.add_argument("--task", choices=["cls", "seg"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--condition", type=str, default="night")
    parser.add_argument("--normal-split", type=str, default="normal")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--num-bands", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cls-epochs", type=int, default=10)
    parser.add_argument("--cls-lr", type=float, default=1e-3)
    parser.add_argument("--cls-hidden-dim", type=int, default=256)
    parser.add_argument("--cls-dropout", type=float, default=0.1)
    parser.add_argument(
        "--cls-domain",
        choices=["freq", "spatial"],
        default="freq",
        help="Use pooled rejected frequency features or IDST-restored spatial features.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/nightadapter")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-night-adapter",
        action="store_true",
        help="Force model.night_adapter.enabled=false before building the model.",
    )
    return parser.parse_args()


def _split_csv_path(config: dict[str, Any], split: str) -> Path:
    data_config = config["data"]
    root = Path(data_config.get("root", "."))
    split_dir = Path(data_config.get("split_dir", "data/splits"))
    if not split_dir.is_absolute():
        split_dir = root / split_dir
    return split_dir / f"{split}.csv"


def _available_conditions(dataset: Any) -> list[str]:
    if not hasattr(dataset, "samples"):
        return []
    return sorted({str(sample.get("condition", "unknown")) for sample in dataset.samples})


def _filter_by_condition(dataset: Any, condition: str, split: str) -> Subset:
    if not hasattr(dataset, "samples"):
        raise ValueError("Dataset does not expose .samples, so condition filtering is unavailable.")
    indices = [
        idx
        for idx, sample in enumerate(dataset.samples)
        if str(sample.get("condition", "unknown")) == condition
    ]
    if len(indices) == 0:
        raise ValueError(
            f"No samples found for condition={condition!r} in split={split!r}. "
            f"Available conditions: {_available_conditions(dataset)}"
        )
    return Subset(dataset, indices)


def _filter_normal_by_split(dataset: Any, target_split: str) -> Subset:
    if not hasattr(dataset, "samples"):
        raise ValueError("Dataset does not expose .samples, so split filtering is unavailable.")
    indices = []
    for idx, sample in enumerate(dataset.samples):
        if str(sample.get("split", "")) != target_split:
            continue
        if not str(sample.get("label_path", "")).strip():
            continue
        indices.append(idx)
    if len(indices) == 0:
        available = sorted({str(sample.get("split", "")) for sample in dataset.samples})
        raise ValueError(
            f"No normal samples found for split={target_split!r}. "
            f"Available normal split values: {available}"
        )
    return Subset(dataset, indices)


def build_condition_dataset(
    config: dict[str, Any],
    split: str,
    condition: str,
    normal_split: str = "normal",
) -> Dataset:
    if condition == "normal" and _split_csv_path(config, normal_split).exists():
        dataset = build_dataset(config, split=normal_split)
        return _filter_normal_by_split(dataset, target_split=split)

    dataset = build_dataset(config, split=split)
    return _filter_by_condition(dataset, condition=condition, split=split)


def build_binary_loader(
    config: dict[str, Any],
    split: str,
    condition: str,
    normal_split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    normal_dataset = build_condition_dataset(config, split, "normal", normal_split)
    night_dataset = build_condition_dataset(config, split, condition, normal_split)
    dataset = ConcatDataset(
        [
            BinaryConditionDataset(normal_dataset, label=0),
            BinaryConditionDataset(night_dataset, label=1),
        ]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_seg_loader(
    config: dict[str, Any],
    split: str,
    condition: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = build_condition_dataset(config, split, condition)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_model(
    config: dict[str, Any],
    checkpoint: str | None,
    device: torch.device,
    disable_night_adapter: bool,
) -> nn.Module:
    model_config = config.setdefault("model", {})
    if disable_night_adapter:
        night_adapter_config = model_config.setdefault("night_adapter", {})
        night_adapter_config["enabled"] = False

    model = build_model(config).to(device)
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing checkpoint keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(
                f"Warning: unexpected checkpoint keys: "
                f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
            )
    model.eval()
    return model


def pool_feature_after_rejection(
    feature: torch.Tensor,
    rejected_band: int | None,
    num_bands: int,
    domain: str,
) -> torch.Tensor:
    flat, spatial_size = flatten_spatial(feature)
    freq = dst_channel(flat)
    freq = reject_frequency_band(freq, rejected_band=rejected_band, num_bands=num_bands)

    if domain == "freq":
        pooled = freq.mean(dim=-1)
    elif domain == "spatial":
        restored = idst_channel(freq)
        restored = unflatten_spatial(restored, spatial_size)
        pooled = restored.flatten(2).mean(dim=-1)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return pooled


@torch.no_grad()
def extract_binary_features(
    model: nn.Module,
    loader: DataLoader,
    stage: int,
    rejected_band: int | None,
    num_bands: int,
    domain: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    feature_idx = int(stage) - 1

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["binary_label"].to(device, non_blocking=True)
        stage_features = model.extract_encoder_features(images)
        pooled = pool_feature_after_rejection(
            stage_features[feature_idx],
            rejected_band=rejected_band,
            num_bands=num_bands,
            domain=domain,
        )
        features_list.append(pooled.cpu())
        labels_list.append(labels.cpu())

    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def train_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    classifier = BandMLP(
        in_dim=int(train_x.shape[1]),
        hidden_dim=int(args.cls_hidden_dim),
        dropout=float(args.cls_dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=float(args.cls_lr), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    for _ in range(int(args.cls_epochs)):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        train_pred = classifier(train_x).argmax(dim=1)
        val_pred = classifier(val_x).argmax(dim=1)
    train_acc = (train_pred == train_y).float().mean().item()
    val_acc = (val_pred == val_y).float().mean().item()
    return {"source_acc": float(train_acc), "target_acc": float(val_acc)}


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], path: Path, metric_name: str, title: str) -> None:
    labels = [str(row["rejected_band"]) for row in rows]
    values = [float(row[metric_name]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xlabel("Rejected band")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_cls(args: argparse.Namespace, config: dict[str, Any], model: nn.Module, device: torch.device) -> None:
    train_config = config.get("train", {})
    batch_size = int(args.batch_size or train_config.get("batch_size", 2))
    num_workers = int(args.num_workers if args.num_workers is not None else train_config.get("num_workers", 4))

    train_loader = build_binary_loader(
        config,
        split=args.train_split,
        condition=args.condition,
        normal_split=args.normal_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    val_loader = build_binary_loader(
        config,
        split=args.val_split,
        condition=args.condition,
        normal_split=args.normal_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    output_dir = ensure_dir(Path(args.output_dir) / "band_rejection_cls")
    rejected_bands: list[int | None] = [None] + list(range(int(args.num_bands)))

    for stage in args.stages:
        rows: list[dict[str, Any]] = []
        baseline_target_acc: float | None = None
        print(f"[CLS] Stage {stage}")
        for rejected_band in rejected_bands:
            label = "none" if rejected_band is None else f"band{rejected_band}"
            print(f"  extracting features: rejected_band={label}")
            train_x, train_y = extract_binary_features(
                model,
                train_loader,
                stage=stage,
                rejected_band=rejected_band,
                num_bands=args.num_bands,
                domain=args.cls_domain,
                device=device,
            )
            val_x, val_y = extract_binary_features(
                model,
                val_loader,
                stage=stage,
                rejected_band=rejected_band,
                num_bands=args.num_bands,
                domain=args.cls_domain,
                device=device,
            )
            metrics = train_classifier(train_x, train_y, val_x, val_y, args, device)
            if rejected_band is None:
                baseline_target_acc = metrics["target_acc"]
            target_delta = (
                metrics["target_acc"] - baseline_target_acc
                if baseline_target_acc is not None
                else 0.0
            )
            row = {
                "task": "cls",
                "stage": int(stage),
                "rejected_band": label,
                "source_acc": metrics["source_acc"],
                "target_acc": metrics["target_acc"],
                "target_acc_delta": float(target_delta),
                "domain": args.cls_domain,
            }
            rows.append(row)
            print(
                f"    source_acc={row['source_acc']:.4f}, "
                f"target_acc={row['target_acc']:.4f}, "
                f"delta={row['target_acc_delta']:+.4f}"
            )

        save_csv(rows, output_dir / f"stage{stage}_cls.csv")
        plot_metric(
            rows,
            output_dir / f"stage{stage}_target_acc.png",
            metric_name="target_acc",
            title=f"Stage {stage} band rejection classification target accuracy",
        )
        plot_metric(
            rows,
            output_dir / f"stage{stage}_target_delta.png",
            metric_name="target_acc_delta",
            title=f"Stage {stage} band rejection classification target delta",
        )


def run_seg(args: argparse.Namespace, config: dict[str, Any], model: nn.Module, device: torch.device) -> None:
    eval_config = config.get("evaluate", {})
    train_config = config.get("train", {})
    batch_size = int(args.batch_size or eval_config.get("batch_size", train_config.get("batch_size", 2)))
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else eval_config.get("num_workers", train_config.get("num_workers", 4))
    )
    loader = build_seg_loader(
        config,
        split=args.val_split,
        condition=args.condition,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    criterion = build_loss(config).to(device)
    metric = SegmentationMetric(
        num_classes=int(config["data"]["num_classes"]),
        ignore_index=int(config["data"].get("ignore_index", 255)),
        device=device,
    )
    output_dir = ensure_dir(Path(args.output_dir) / "band_rejection_seg")
    rejected_bands: list[int | None] = [None] + list(range(int(args.num_bands)))

    for stage in args.stages:
        rows: list[dict[str, Any]] = []
        baseline_miou: float | None = None
        print(f"[SEG] Stage {stage}")
        for rejected_band in rejected_bands:
            label = "none" if rejected_band is None else f"band{rejected_band}"
            if rejected_band is None:
                model.clear_band_rejection()
            else:
                model.set_band_rejection(stage=stage, band=rejected_band, num_bands=args.num_bands)

            metric.reset()
            losses: list[float] = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    logits = model(images)
                    loss = criterion(logits, masks)
                    losses.append(float(loss.item()))
                    metric.update(logits, masks)
            result = metric.compute()
            miou = float(result["miou"])
            if rejected_band is None:
                baseline_miou = miou
            miou_delta = miou - baseline_miou if baseline_miou is not None else 0.0
            row = {
                "task": "seg",
                "stage": int(stage),
                "rejected_band": label,
                "loss": float(sum(losses) / max(1, len(losses))),
                "miou": miou,
                "miou_delta": float(miou_delta),
            }
            rows.append(row)
            print(f"  {label}: mIoU={miou:.4f}, delta={miou_delta:+.4f}")

        model.clear_band_rejection()
        save_csv(rows, output_dir / f"stage{stage}_seg.csv")
        plot_metric(
            rows,
            output_dir / f"stage{stage}_miou.png",
            metric_name="miou",
            title=f"Stage {stage} band rejection segmentation mIoU",
        )
        plot_metric(
            rows,
            output_dir / f"stage{stage}_miou_delta.png",
            metric_name="miou_delta",
            title=f"Stage {stage} band rejection segmentation mIoU delta",
        )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    config = load_config(args.config)
    config = copy.deepcopy(config)
    device = torch.device(args.device) if args.device is not None else get_device()
    print(f"Using device: {device}")

    model = load_model(
        config=config,
        checkpoint=args.checkpoint,
        device=device,
        disable_night_adapter=bool(args.disable_night_adapter),
    )

    if args.task == "cls":
        run_cls(args, config, model, device)
    elif args.task == "seg":
        run_seg(args, config, model, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
