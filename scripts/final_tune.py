#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final refinement tuning for SegFormer with Optuna.

This script is intended for the final model search of
Adverse-Weather-Segmentation.

Key assumptions:
- Base config: configs/final.yaml
- Fixed input size: 1024 x 512, or whatever is already written in the config.
- Fixed batch size: 8
- Fixed loss / augmentation / enhancement settings from the base config.
- Tuned parameters only:
  lr, head_lr_mult, dropout, drop_path_rate, weight_decay,
  warmup_epochs, scheduler_power
  plus frozen constants freeze_mode=["full"], encoder_lr_mult=[1.0].

Example:
    python scripts/final_tune.py \
      --config configs/final.yaml \
      --output-dir outputs/tuning/segformer_final_refine \
      --storage sqlite:///outputs/tuning/segformer_final_refine/optuna.db \
      --study-name segformer_final_refine \
      --n-trials 50 \
      --max-epochs-per-trial 40 \
      --device cuda:0 \
      --disable-wandb
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import yaml


FINAL_BATCH_SIZE = 8

DEFAULT_OUTPUT_DIR = "outputs/tuning/segformer_final_refine"
DEFAULT_STUDY_NAME = "segformer_final_refine"

# DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
#     "freeze_mode": ["full"],
#     "lr": [0.00011, 0.00012, 0.00013, 0.00014],
#     "encoder_lr_mult": [1.0],
#     "head_lr_mult": [3.5, 4.0, 4.5],
#     "dropout": [0.03, 0.05],
#     "drop_path_rate": [0.15, 0.18, 0.20],
#     "weight_decay": [0.005, 0.01, 0.02],
#     "warmup_epochs": [3, 5],
#     "scheduler_power": [0.9, 1.0],
# }

DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "freeze_mode": ["full"],
    "lr": [0.00005, 0.00006, 0.00009, 0.00012],
    "encoder_lr_mult": [1.0],
    "head_lr_mult": [4.0, 8.0, 12.0],
    "dropout": [0.05, 0.1],
    "drop_path_rate": [0.1, 0.15, 0.2],
    "weight_decay": [0.005, 0.01, 0.02],
    "warmup_epochs": [3, 5],
    "scheduler_power": [0.9, 1.0],
}

DEFAULT_PRETRAINED_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# These are intentionally not tunable in final refinement.
FORBIDDEN_SEARCH_KEYS = {
    "batch_size",
    "input_width",
    "input_height",
    "loss",
    "loss_name",
    "augmentation",
    "augmentation_name",
    "enhancement",
    "enhancement_name",
    "gamma",
    "apply_to",
    "apply_conditions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Final Optuna refinement for SegFormer with batch_size={FINAL_BATCH_SIZE}."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final.yaml",
        help="Base final config. Loss/augmentation/enhancement are kept fixed.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="",
        choices=["", "fog", "rain", "snow", "night"],
        help="Optional weather condition. Empty string means all conditions.",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--direction", type=str, default=None, choices=["maximize", "minimize"])
    parser.add_argument(
        "--resume-study",
        action="store_true",
        help=(
            "Reuse an existing Optuna study. Do not use this when changing "
            "categorical choices from an older study."
        ),
    )
    parser.add_argument(
        "--max-epochs-per-trial",
        type=int,
        default=None,
        help="Override train.epochs for each tuning trial.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to optimize. Defaults to tuning.metric or best_miou.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for tuning artifacts. Defaults to tuning.output_dir or "
            f"{DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional device argument passed to awseg.train/evaluate, e.g. cuda:0.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run awseg.evaluate after each training trial before reading the metric.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Force wandb.enabled=false in each trial config.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed for Optuna TPESampler. Defaults to tuning.sampler_seed or 42.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML config: {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def recursive_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            recursive_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def get_search_space(config: dict[str, Any]) -> dict[str, list[Any]]:
    tuning_config = config.get("tuning", {})
    user_space = tuning_config.get("search_space", {})

    search_space = copy.deepcopy(DEFAULT_SEARCH_SPACE)
    if isinstance(user_space, dict):
        recursive_update(search_space, user_space)

    forbidden = sorted(set(search_space) & FORBIDDEN_SEARCH_KEYS)
    if forbidden:
        raise ValueError(
            "These keys must not be in tuning.search_space for final_tune.py: "
            + ", ".join(forbidden)
        )

    for name, values in search_space.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Search space for {name!r} must be a non-empty list.")

    return search_space


def suggest_categorical(trial: optuna.Trial, name: str, values: list[Any]) -> Any:
    return trial.suggest_categorical(name, values)


def build_condition_args(condition: str) -> list[str]:
    if not condition:
        return []
    return ["--condition", condition]


def maybe_append_device(command: list[str], device: str) -> list[str]:
    if device:
        command.extend(["--device", device])
    return command


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command: " + " ".join(command) + "\n\n")
        log_file.flush()

        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. See log: {log_path}"
        )


def train_trial(
    config_path: Path,
    result_dir: Path,
    condition: str,
    device: str,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "awseg.train",
        "--config",
        str(config_path),
        "--result-dir",
        str(result_dir),
        *build_condition_args(condition),
    ]
    maybe_append_device(command, device)
    run_command(command, log_path)


def evaluate_trial(
    config_path: Path,
    checkpoint_path: Path,
    result_dir: Path,
    condition: str,
    device: str,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "awseg.evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "val",
        "--result-dir",
        str(result_dir),
        *build_condition_args(condition),
    ]
    maybe_append_device(command, device)
    run_command(command, log_path)


def find_json_files(result_dir: Path, condition: str) -> list[Path]:
    preferred_names: list[str] = []

    if condition:
        preferred_names.extend(
            [
                f"train_{condition}.json",
                f"eval_val_{condition}.json",
                f"eval_{condition}.json",
            ]
        )

    preferred_names.extend(
        [
            "train.json",
            "train_val.json",
            "eval_val.json",
            "eval.json",
            "metrics.json",
            "summary.json",
        ]
    )

    files: list[Path] = []
    for name in preferred_names:
        path = result_dir / name
        if path.exists():
            files.append(path)

    files.extend(sorted(p for p in result_dir.rglob("*.json") if p not in files))
    return files


def flatten_metrics(data: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_metrics(value, name))
    elif isinstance(data, (int, float)):
        metrics[prefix] = float(data)

    return metrics


def read_metric(result_dir: Path, metric_name: str, condition: str) -> tuple[float, Path, str]:
    json_files = find_json_files(result_dir, condition)
    if not json_files:
        raise FileNotFoundError(f"No metric JSON files found in: {result_dir}")

    aliases = [
        metric_name,
        metric_name.lower(),
        metric_name.upper(),
        "best_miou",
        "val_miou",
        "miou",
        "mIoU",
        "mean_iou",
        "mean_IoU",
    ]

    seen: set[str] = set()
    ordered_aliases: list[str] = []
    for alias in aliases:
        if alias not in seen:
            ordered_aliases.append(alias)
            seen.add(alias)

    available: dict[str, list[str]] = {}

    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = flatten_metrics(data)
        available[str(path)] = sorted(metrics.keys())

        for alias in ordered_aliases:
            if alias in metrics:
                return float(metrics[alias]), path, alias

        # Also allow suffix matching such as metrics.val_miou.
        for alias in ordered_aliases:
            for key, value in metrics.items():
                if key.endswith(f".{alias}") or key.endswith(alias):
                    return float(value), path, key

    raise KeyError(
        f"Metric {metric_name!r} not found in JSON files under {result_dir}.\n"
        f"Available metrics: {json.dumps(available, indent=2, ensure_ascii=False)}"
    )


def apply_trial_config(
    base_config: dict[str, Any],
    trial: optuna.Trial,
    trial_dir: Path,
    result_dir: Path,
    analysis_dir: Path,
    visualization_dir: Path,
    condition: str,
    max_epochs_per_trial: int | None,
    disable_wandb: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    search_space = get_search_space(config)

    model_config = config.setdefault("model", {})
    optimizer_config = config.setdefault("optimizer", {})
    scheduler_config = config.setdefault("scheduler", {})
    train_config = config.setdefault("train", {})
    checkpoint_config = config.setdefault("checkpoint", {})
    evaluate_config = config.setdefault("evaluate", {})
    output_config = config.setdefault("output", {})
    experiment_config = config.setdefault("experiment", {})
    wandb_config = config.setdefault("wandb", {})

    trial_name = f"trial_{trial.number:04d}"
    run_name = trial_name if not condition else f"{trial_name}_{condition}"

    # Final tuning script constraints.
    # Do not touch data.input_width / data.input_height.
    # Force batch size to the final intended value.
    train_config["batch_size"] = FINAL_BATCH_SIZE

    # Force SegFormer, but keep the final config pretrained_name if it already exists.
    model_config["name"] = "segformer"
    model_config.setdefault("pretrained_name", DEFAULT_PRETRAINED_NAME)

    if "freeze_mode" in search_space:
        model_config["freeze_mode"] = str(
            suggest_categorical(trial, "freeze_mode", search_space["freeze_mode"])
        )

    if "dropout" in search_space:
        model_config["dropout"] = float(
            suggest_categorical(trial, "dropout", search_space["dropout"])
        )

    if "drop_path_rate" in search_space:
        model_config["drop_path_rate"] = float(
            suggest_categorical(trial, "drop_path_rate", search_space["drop_path_rate"])
        )

    if "lr" in search_space:
        optimizer_config["lr"] = float(
            suggest_categorical(trial, "lr", search_space["lr"])
        )

    if "encoder_lr_mult" in search_space:
        optimizer_config["encoder_lr_mult"] = float(
            suggest_categorical(
                trial,
                "encoder_lr_mult",
                search_space["encoder_lr_mult"],
            )
        )

    if "head_lr_mult" in search_space:
        optimizer_config["head_lr_mult"] = float(
            suggest_categorical(trial, "head_lr_mult", search_space["head_lr_mult"])
        )

    if "weight_decay" in search_space:
        optimizer_config["weight_decay"] = float(
            suggest_categorical(trial, "weight_decay", search_space["weight_decay"])
        )

    # Prefer multiplier-based LR in the integrated train.py.
    optimizer_config.pop("encoder_lr", None)
    optimizer_config.pop("head_lr", None)

    if "warmup_epochs" in search_space:
        scheduler_config["warmup_epochs"] = int(
            suggest_categorical(
                trial,
                "warmup_epochs",
                search_space["warmup_epochs"],
            )
        )

    if "scheduler_power" in search_space:
        scheduler_config["power"] = float(
            suggest_categorical(
                trial,
                "scheduler_power",
                search_space["scheduler_power"],
            )
        )

    if max_epochs_per_trial is not None:
        train_config["epochs"] = int(max_epochs_per_trial)

        # Keep cosine scheduler length consistent when applicable.
        scheduler_name = str(scheduler_config.get("name", "")).lower()
        if scheduler_name in {"cosine", "cosine_annealing"}:
            scheduler_config["T_max"] = int(max_epochs_per_trial)

    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_config["save_dir"] = str(checkpoint_dir)
    checkpoint_config.setdefault("save_best_name", "best_miou.pth")
    checkpoint_config.setdefault("save_last_name", "last.pth")

    evaluate_config["output_dir"] = str(result_dir)

    output_config["checkpoint_dir"] = str(checkpoint_dir)
    output_config["result_dir"] = str(result_dir)
    output_config["analysis_dir"] = str(analysis_dir)
    output_config["visualization_dir"] = str(visualization_dir)
    output_config["log_dir"] = str(trial_dir / "logs")

    experiment_config["name"] = run_name
    experiment_config["group"] = "final_refine_tuning"

    if disable_wandb or bool(config.get("tuning", {}).get("disable_wandb", True)):
        wandb_config["enabled"] = False
    elif wandb_config.get("enabled", False):
        wandb_config["run_name"] = f"final_tune_{run_name}"
        tags = list(wandb_config.get("tags", []) or [])
        for tag in ["final_tuning", "segformer", f"bs{FINAL_BATCH_SIZE}", run_name]:
            if tag not in tags:
                tags.append(tag)
        wandb_config["tags"] = tags

    return config


def main() -> None:
    args = parse_args()

    base_config_path = Path(args.config)
    base_config = load_yaml(base_config_path)
    tuning_config = base_config.get("tuning", {})

    output_dir = Path(
        args.output_dir
        or tuning_config.get(
            "output_dir",
            DEFAULT_OUTPUT_DIR,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "configs"
    result_root = output_dir / "results"
    log_dir = output_dir / "logs"
    trial_root = output_dir / "trials"
    analysis_root = output_dir / "analysis"
    visualization_root = output_dir / "visualizations"

    n_trials = int(args.n_trials or tuning_config.get("n_trials", 50))

    max_epochs_per_trial = args.max_epochs_per_trial
    if max_epochs_per_trial is None:
        max_epochs_per_trial = tuning_config.get("max_epochs_per_trial", 40)
    if max_epochs_per_trial is not None:
        max_epochs_per_trial = int(max_epochs_per_trial)

    study_name = str(
        args.study_name
        or tuning_config.get("study_name", DEFAULT_STUDY_NAME)
    )
    storage = str(
        args.storage
        or tuning_config.get("storage", f"sqlite:///{output_dir / 'optuna.db'}")
    )
    direction = str(args.direction or tuning_config.get("direction", "maximize"))
    metric_name = str(args.metric or tuning_config.get("metric", "best_miou"))
    sampler_seed = int(
        args.sampler_seed
        if args.sampler_seed is not None
        else tuning_config.get("sampler_seed", 42)
    )

    search_space = get_search_space(base_config)

    data_config = base_config.get("data", {})
    input_width = data_config.get("input_width", "config/default")
    input_height = data_config.get("input_height", "config/default")

    print("============================================================")
    print("Final SegFormer Optuna tuning")
    print("============================================================")
    print(f"Config              : {base_config_path}")
    print(f"Condition           : {args.condition or 'all'}")
    print(f"Output dir          : {output_dir}")
    print(f"Study name          : {study_name}")
    print(f"Storage             : {storage}")
    print(f"Direction           : {direction}")
    print(f"Metric              : {metric_name}")
    print(f"N trials            : {n_trials}")
    print(f"Timeout             : {args.timeout}")
    print(f"Max epochs per trial: {max_epochs_per_trial}")
    print(f"Run eval per trial  : {args.run_eval}")
    print(f"Device              : {args.device or 'config/default'}")
    print(f"Sampler seed        : {sampler_seed}")
    print(f"Fixed batch size    : {FINAL_BATCH_SIZE}")
    print(f"Fixed input size    : {input_width} x {input_height}")
    print("Search space:")
    print(json.dumps(search_space, indent=2, ensure_ascii=False))
    print("============================================================")

    def objective(trial: optuna.Trial) -> float:
        trial_name = f"trial_{trial.number:04d}"

        trial_dir = trial_root / trial_name
        trial_result_dir = result_root / trial_name
        trial_analysis_dir = analysis_root / trial_name
        trial_visualization_dir = visualization_root / trial_name
        trial_config_path = config_dir / f"{trial_name}.yaml"
        train_log_path = log_dir / f"{trial_name}_train_{now()}.log"
        eval_log_path = log_dir / f"{trial_name}_eval_{now()}.log"

        trial_config = apply_trial_config(
            base_config=base_config,
            trial=trial,
            trial_dir=trial_dir,
            result_dir=trial_result_dir,
            analysis_dir=trial_analysis_dir,
            visualization_dir=trial_visualization_dir,
            condition=args.condition,
            max_epochs_per_trial=max_epochs_per_trial,
            disable_wandb=args.disable_wandb,
        )
        save_yaml(trial_config, trial_config_path)

        print("")
        print("------------------------------------------------------------")
        print(f"[{trial_name}] params")
        print(json.dumps(trial.params, indent=2, ensure_ascii=False))
        print(f"[{trial_name}] config: {trial_config_path}")
        print("------------------------------------------------------------")

        train_trial(
            config_path=trial_config_path,
            result_dir=trial_result_dir,
            condition=args.condition,
            device=args.device,
            log_path=train_log_path,
        )

        checkpoint_name = trial_config.get("checkpoint", {}).get(
            "save_best_name",
            "best_miou.pth",
        )
        checkpoint_path = Path(trial_config["checkpoint"]["save_dir"]) / checkpoint_name

        if args.run_eval:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

            evaluate_trial(
                config_path=trial_config_path,
                checkpoint_path=checkpoint_path,
                result_dir=trial_result_dir,
                condition=args.condition,
                device=args.device,
                log_path=eval_log_path,
            )

        value, metric_file, metric_key = read_metric(
            result_dir=trial_result_dir,
            metric_name=metric_name,
            condition=args.condition,
        )

        trial.set_user_attr("config_path", str(trial_config_path))
        trial.set_user_attr("result_dir", str(trial_result_dir))
        trial.set_user_attr("checkpoint_path", str(checkpoint_path))
        trial.set_user_attr("metric_file", str(metric_file))
        trial.set_user_attr("metric_key", metric_key)
        trial.set_user_attr("batch_size", FINAL_BATCH_SIZE)
        trial.set_user_attr("input_width", input_width)
        trial.set_user_attr("input_height", input_height)

        print(f"[{trial_name}] {metric_key}: {value:.6f} ({metric_file})")
        return value

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=bool(args.resume_study),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=args.timeout,
    )

    trials_csv_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    best_trial = study.best_trial
    best_config_path = Path(best_trial.user_attrs.get("config_path", ""))
    best_config_copy_path = output_dir / "best_config.yaml"

    if best_config_path.exists():
        shutil.copy2(best_config_path, best_config_copy_path)

    best_summary = {
        "study_name": study.study_name,
        "direction": direction,
        "metric": metric_name,
        "best_trial_number": best_trial.number,
        "best_value": study.best_value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "base_config": str(base_config_path),
        "condition": args.condition or "all",
        "output_dir": str(output_dir),
        "trials_csv": str(trials_csv_path),
        "best_config": str(best_config_copy_path) if best_config_copy_path.exists() else None,
        "fixed_batch_size": FINAL_BATCH_SIZE,
        "fixed_input_width": input_width,
        "fixed_input_height": input_height,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    best_json_path = output_dir / "best_trial.json"
    save_json(best_summary, best_json_path)

    print("")
    print("============================================================")
    print("Best trial")
    print("============================================================")
    print(json.dumps(best_summary, indent=2, ensure_ascii=False))
    print(f"Saved trials CSV : {trials_csv_path}")
    print(f"Saved best JSON  : {best_json_path}")
    if best_config_copy_path.exists():
        print(f"Saved best config: {best_config_copy_path}")
    print("============================================================")


if __name__ == "__main__":
    main()