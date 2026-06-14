#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot semantic segmentation result summaries.")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--result-root", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations/plots")
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument("--history-file", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_result_dir(args: argparse.Namespace) -> Path:
    if args.experiment:
        name = args.experiment
    elif args.group:
        name = args.group
    else:
        name = "baseline"
    return Path(args.result_root) / name


def find_eval_file(args: argparse.Namespace, result_dir: Path) -> Optional[Path]:
    if args.eval_file:
        p = Path(args.eval_file)
        return p if p.exists() else None
    candidates = sorted(result_dir.glob("eval_*.json"))
    if not candidates:
        return None
    return candidates[0]


def find_history_file(args: argparse.Namespace, result_dir: Path) -> Optional[Path]:
    if args.history_file:
        p = Path(args.history_file)
        return p if p.exists() else None
    candidates = sorted(result_dir.glob("train_history*.json"))
    if not candidates:
        return None
    return candidates[0]


def extract_main_summary(eval_data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(eval_data.get("main"), dict):
        return eval_data["main"]
    return eval_data


def plot_overall_and_condition_miou(eval_path: Path, output_dir: Path, dpi: int) -> None:
    data = load_json(eval_path)
    main = extract_main_summary(data)
    labels = ["overall"]
    values = [float(main.get("miou", 0.0))]
    condition_results = main.get("condition_results", {}) or {}
    for condition in sorted(condition_results.keys()):
        labels.append(condition)
        values.append(float(condition_results[condition].get("miou", 0.0)))
    if data.get("normal") is not None:
        labels.append("normal")
        values.append(float(data["normal"].get("miou", 0.0)))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.5))
    ax.bar(labels, values)
    ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    ax.set_ylabel("mIoU")
    ax.set_title("Overall and condition-wise mIoU")
    ax.tick_params(axis="x", rotation=30)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_condition_miou.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_class_iou(eval_path: Path, output_dir: Path, dpi: int) -> None:
    data = load_json(eval_path)
    main = extract_main_summary(data)
    class_iou = main.get("class_iou", {}) or {}
    if not class_iou:
        return
    labels = list(class_iou.keys())
    values = [float(class_iou[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.35)))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlim(0, max(1.0, max(values, default=0.0) * 1.15))
    ax.set_xlabel("IoU")
    ax.set_title("Class-wise IoU")
    fig.tight_layout()
    fig.savefig(output_dir / "class_iou.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_train_history(history_path: Path, output_dir: Path, dpi: int) -> None:
    history = load_json(history_path)
    if not isinstance(history, list) or len(history) == 0:
        return
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]
    train_loss = [float(row.get("train_loss", 0.0)) for row in history]
    val_loss = [float(row.get("val_loss", 0.0)) for row in history]
    train_miou = [float(row.get("train_miou", 0.0)) for row in history]
    val_miou = [float(row.get("val_miou", 0.0)) for row in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="Train loss")
    ax.plot(epochs, val_loss, marker="o", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_miou, marker="o", label="Train mIoU")
    ax.plot(epochs, val_miou, marker="o", label="Val mIoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Training and validation mIoU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "miou_curve.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = resolve_result_dir(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = find_eval_file(args, result_dir)
    history_path = find_history_file(args, result_dir)

    jobs = []
    if eval_path is not None:
        jobs.append(("overall_condition_miou", lambda: plot_overall_and_condition_miou(eval_path, output_dir, args.dpi)))
        jobs.append(("class_iou", lambda: plot_class_iou(eval_path, output_dir, args.dpi)))
    else:
        print(f"[WARN] No eval_*.json found in {result_dir}. Skipping evaluation plots.")

    if history_path is not None:
        jobs.append(("train_history", lambda: plot_train_history(history_path, output_dir, args.dpi)))
    else:
        print(f"[WARN] No train_history*.json found in {result_dir}. Skipping training curve plots.")

    for name, job in tqdm(
        jobs,
        total=len(jobs),
        desc="Plot results",
        dynamic_ncols=True,
        mininterval=2,
        file=sys.stdout,
        leave=True,
    ):
        job()

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
