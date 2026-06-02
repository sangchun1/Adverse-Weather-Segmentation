#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate plots from AWSeg evaluation JSON files.

Default input:
    outputs/results/

Default output:
    outputs/plots/

Examples:
    python scripts/plot_results.py
    python scripts/plot_results.py --conditions fog rain snow night
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_OUTPUT_DIR = Path("outputs/plots")
DEFAULT_RESULTS_DIR = Path("outputs/results")
DEFAULT_BASELINE_JSON = Path("outputs/results/baseline/eval_val.json")


DEFAULT_CLASS_ORDER = [
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


METHOD_NAME_MAP = {
    "": "No Enhancement",
    "clahe": "CLAHE",
    "gamma": "Gamma",
    "gamma_clahe": "Gamma+CLAHE",
    "retinexformer": "RetinexFormer",
    "sci": "SCI",
    "zero_dce": "Zero-DCE",
    "zero-dce": "Zero-DCE",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_eval_record(data: dict[str, Any], key: str = "main") -> dict[str, Any]:
    """Return the metric-containing part of an eval JSON.

    Supported formats:
        1. Flat format:
           {"miou": ..., "class_iou": ..., "condition_results": ...}

        2. Combined format:
           {"main": {"miou": ...}, "normal": {...}}
    """
    if key in data and isinstance(data[key], dict):
        return data[key]

    return data


def get_miou(data: dict[str, Any]) -> float:
    record = get_eval_record(data)

    for key in ("miou", "mIoU", "mean_iou", "best_miou"):
        if key in record and record[key] is not None:
            return float(record[key])

    raise KeyError("JSON에서 miou 값을 찾지 못했습니다.")


def get_num_samples(data: dict[str, Any]) -> int | None:
    record = get_eval_record(data)
    value = record.get("num_samples")

    if value is None:
        return None

    return int(value)


def get_condition(data: dict[str, Any]) -> str | None:
    record = get_eval_record(data)
    return record.get("condition")


def get_condition_results(data: dict[str, Any]) -> dict[str, Any]:
    record = get_eval_record(data)
    result = record.get("condition_results", {})

    if not isinstance(result, dict):
        return {}

    return result


def get_class_iou(data: dict[str, Any]) -> dict[str, float]:
    record = get_eval_record(data)
    class_iou = record.get("class_iou", {})

    if not isinstance(class_iou, dict):
        return {}

    clean: dict[str, float] = {}

    for key, value in class_iou.items():
        if value is None:
            continue
        clean[str(key)] = float(value)

    return clean


def normalize_method_name(folder_name: str, condition: str) -> str:
    if folder_name == condition:
        suffix = ""
    elif folder_name.startswith(condition + "_"):
        suffix = folder_name[len(condition) + 1 :]
    else:
        suffix = folder_name

    if suffix in METHOD_NAME_MAP:
        return METHOD_NAME_MAP[suffix]

    return suffix.replace("_", " ").replace("-", " ").title()


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_baseline(
    baseline_json: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    baseline_output_dir = ensure_dir(output_dir / "baseline")
    data = load_json(baseline_json)

    rows: list[dict[str, Any]] = [
        {
            "name": "Overall",
            "miou": get_miou(data),
            "n": get_num_samples(data),
            "group": "Overall",
        }
    ]

    condition_results = get_condition_results(data)
    preferred_order = ["fog", "rain", "snow", "night"]
    remaining = [c for c in condition_results.keys() if c not in preferred_order]

    for condition in preferred_order + remaining:
        if condition not in condition_results:
            continue

        result = condition_results[condition]
        rows.append(
            {
                "name": condition.capitalize(),
                "miou": float(result["miou"]),
                "n": result.get("num_images"),
                "group": "Weather",
            }
        )

    df = pd.DataFrame(rows)

    set_style()
    palette = {
        "Overall": "#4C72B0",
        "Weather": "#DD8452",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=df,
        x="name",
        y="miou",
        hue="group",
        dodge=False,
        palette=palette,
        ax=ax,
    )

    ax.set_title("Baseline mIoU by Adverse Weather Condition", pad=18, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, min(1.0, float(df["miou"].max()) + 0.15))
    ax.legend(title="", loc="upper right", frameon=True)

    for patch, (_, row) in zip(ax.patches[: len(df)], df.iterrows()):
        height = patch.get_height()
        n_text = "" if pd.isna(row["n"]) else f"\n(n={int(row['n'])})"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.012,
            f"{row['miou']:.4f}{n_text}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    sns.despine(left=False, bottom=False)
    fig.tight_layout()
    save_figure(
        fig,
        baseline_output_dir / "baseline_miou_by_condition.png",
        dpi=dpi,
    )

    class_iou = get_class_iou(data)

    if not class_iou:
        print("[WARN] baseline class_iou가 없어서 class-wise plot은 생략합니다.")
        return

    class_df = (
        pd.DataFrame([{"class": cls, "iou": iou} for cls, iou in class_iou.items()])
        .sort_values("iou", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.barplot(
        data=class_df,
        x="iou",
        y="class",
        hue="class",
        palette="crest",
        legend=False,
        ax=ax,
    )

    ax.set_title("Baseline Class-wise IoU", pad=18, weight="bold")
    ax.set_xlabel("IoU")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)

    for patch, (_, row) in zip(ax.patches, class_df.iterrows()):
        width = patch.get_width()
        ax.text(
            width + 0.012,
            patch.get_y() + patch.get_height() / 2,
            f"{row['iou']:.4f}",
            va="center",
            fontsize=9,
        )

    sns.despine(left=False, bottom=False)
    fig.tight_layout()
    save_figure(
        fig,
        baseline_output_dir / "baseline_class_iou.png",
        dpi=dpi,
    )

    print(f"[OK] baseline plots saved to: {baseline_output_dir}")


def collect_condition_results(
    results_dir: Path,
    condition: str,
    eval_filename: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {results_dir}")

    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        folder = method_dir.name

        if folder != condition and not folder.startswith(condition + "_"):
            continue

        json_path = method_dir / eval_filename

        if not json_path.exists():
            continue

        data = load_json(json_path)
        json_condition = get_condition(data)

        if json_condition is not None and json_condition != condition:
            continue

        class_iou = get_class_iou(data)

        rows.append(
            {
                "folder": folder,
                "method": normalize_method_name(folder, condition),
                "miou": get_miou(data),
                "num_samples": get_num_samples(data),
                "json_path": str(json_path),
                "class_iou": class_iou,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"{results_dir}에서 prefix='{condition}'인 {eval_filename} 결과를 찾지 못했습니다."
        )

    return pd.DataFrame(rows)


def get_method_palette(
    plot_df: pd.DataFrame,
    baseline_method: str,
    baseline_miou: float,
) -> dict[str, str]:
    palette: dict[str, str] = {}

    for _, row in plot_df.iterrows():
        method = row["method"]
        miou = float(row["miou"])

        if method == baseline_method:
            palette[method] = "#4C72B0"
        elif miou >= baseline_miou:
            palette[method] = "#55A868"
        else:
            palette[method] = "#C44E52"

    return palette


def build_class_matrix(df: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    all_classes: list[str] = []

    for class_iou in df["class_iou"]:
        for cls in class_iou.keys():
            if cls not in all_classes:
                all_classes.append(cls)

    class_order = [c for c in DEFAULT_CLASS_ORDER if c in all_classes]
    class_order += [c for c in all_classes if c not in class_order]

    class_matrix = pd.DataFrame(index=method_order, columns=class_order, dtype=float)

    for _, row in df.iterrows():
        method = row["method"]

        for cls, iou in row["class_iou"].items():
            if cls in class_matrix.columns:
                class_matrix.loc[method, cls] = float(iou)

    return class_matrix


def plot_condition_comparison(
    results_dir: Path,
    condition: str,
    output_dir: Path,
    eval_filename: str = "eval_val.json",
    dpi: int = 300,
) -> None:
    condition_output_dir = ensure_dir(output_dir / condition)
    df = collect_condition_results(results_dir, condition, eval_filename)

    baseline_method = "No Enhancement"

    if baseline_method not in set(df["method"]):
        exact = df[df["folder"] == condition]
        baseline_method = exact.iloc[0]["method"] if len(exact) > 0 else df.iloc[0]["method"]

    baseline_miou = float(df[df["method"] == baseline_method]["miou"].iloc[0])

    summary_df = (
        df.drop(columns=["class_iou"])
        .assign(delta_vs_baseline=lambda x: x["miou"] - baseline_miou)
        .sort_values("miou", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(
        condition_output_dir / f"{condition}_enhancement_summary.csv",
        index=False,
    )

    method_order = summary_df["method"].tolist()
    class_matrix = build_class_matrix(df, method_order)
    class_matrix.to_csv(condition_output_dir / f"{condition}_class_iou_matrix.csv")

    set_style()
    title_condition = condition.capitalize()
    eval_tag = eval_filename.replace(".json", "")

    # 1. mIoU comparison
    plot_df = summary_df.copy()
    palette = get_method_palette(plot_df, baseline_method, baseline_miou)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="method",
        y="miou",
        hue="method",
        palette=palette,
        legend=False,
        ax=ax,
    )

    ax.axhline(
        baseline_miou,
        linestyle="--",
        linewidth=1.5,
        color="#4C72B0",
        label=f"{baseline_method} baseline: {baseline_miou:.4f}",
    )

    ax.set_title(
        f"{title_condition} Enhancement Methods: mIoU Comparison ({eval_tag})",
        pad=18,
        weight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, min(1.0, float(plot_df["miou"].max()) + 0.08))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.006,
            f"{row['miou']:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.tick_params(axis="x", rotation=25)
    sns.despine(left=False, bottom=False)
    fig.tight_layout()
    save_figure(
        fig,
        condition_output_dir / f"{condition}_miou_comparison.png",
        dpi=dpi,
    )

    # 2. delta mIoU
    delta_df = (
        summary_df[summary_df["method"] != baseline_method]
        .copy()
        .sort_values("delta_vs_baseline", ascending=False)
        .reset_index(drop=True)
    )

    if len(delta_df) > 0:
        delta_palette = {
            row["method"]: "#55A868" if row["delta_vs_baseline"] >= 0 else "#C44E52"
            for _, row in delta_df.iterrows()
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=delta_df,
            x="method",
            y="delta_vs_baseline",
            hue="method",
            palette=delta_palette,
            legend=False,
            ax=ax,
        )

        ax.axhline(0, linewidth=1.2, color="black")
        ax.set_title(
            f"{title_condition} Enhancement Methods: ΔmIoU vs {baseline_method}",
            pad=18,
            weight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("ΔmIoU")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for patch, (_, row) in zip(ax.patches, delta_df.iterrows()):
            value = float(row["delta_vs_baseline"])
            offset = 0.003 if value >= 0 else -0.003
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                value + offset,
                f"{value:+.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
            )

        ax.tick_params(axis="x", rotation=25)
        sns.despine(left=False, bottom=False)
        fig.tight_layout()
        save_figure(
            fig,
            condition_output_dir / f"{condition}_miou_delta.png",
            dpi=dpi,
        )

    if class_matrix.empty:
        print(f"[WARN] {condition}: class_iou가 없어서 heatmap은 생략합니다.")
        print(f"[OK] {condition} plots saved to: {condition_output_dir}")
        return

    # 3. class-wise IoU heatmap
    if baseline_method in class_matrix.index:
        heatmap_class_order = (
            class_matrix.loc[baseline_method]
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        heatmap_class_order = class_matrix.columns.tolist()

    heatmap_data = class_matrix.loc[method_order, heatmap_class_order]

    fig, ax = plt.subplots(figsize=(16, max(5, 0.7 * len(method_order))))
    sns.heatmap(
        heatmap_data,
        cmap="magma",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "IoU"},
        ax=ax,
    )

    ax.set_title(
        f"{title_condition} Enhancement Methods: Class-wise IoU Heatmap ({eval_tag})",
        pad=18,
        weight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(
        fig,
        condition_output_dir / f"{condition}_class_iou_heatmap.png",
        dpi=dpi,
    )

    # 4. class-wise delta IoU heatmap
    if baseline_method in class_matrix.index:
        delta_heatmap = class_matrix.sub(class_matrix.loc[baseline_method], axis=1)
        delta_heatmap = delta_heatmap.drop(index=baseline_method, errors="ignore")
        delta_heatmap = delta_heatmap.loc[
            [m for m in method_order if m in delta_heatmap.index],
            heatmap_class_order,
        ]

        if not delta_heatmap.empty:
            fig, ax = plt.subplots(figsize=(16, max(5, 0.7 * len(delta_heatmap))))
            sns.heatmap(
                delta_heatmap,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt="+.2f",
                linewidths=0.3,
                linecolor="white",
                cbar_kws={"label": "ΔIoU"},
                ax=ax,
            )

            ax.set_title(
                f"{title_condition} Enhancement Methods: Class-wise ΔIoU vs {baseline_method}",
                pad=18,
                weight="bold",
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)
            fig.tight_layout()
            save_figure(
                fig,
                condition_output_dir / f"{condition}_class_iou_delta_heatmap.png",
                dpi=dpi,
            )

    print(f"[OK] {condition} plots saved to: {condition_output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline and condition-wise enhancement plots from eval JSON files."
    )

    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--conditions", nargs="+", default=["night"])
    parser.add_argument("--eval-filename", default="eval_val.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-baseline", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if not args.skip_baseline:
        plot_baseline(
            baseline_json=args.baseline_json,
            output_dir=output_dir,
            dpi=args.dpi,
        )

    for condition in args.conditions:
        plot_condition_comparison(
            results_dir=args.results_dir,
            condition=condition,
            output_dir=output_dir,
            eval_filename=args.eval_filename,
            dpi=args.dpi,
        )

    print(f"[DONE] All plots generated under: {output_dir}")


if __name__ == "__main__":
    main()
