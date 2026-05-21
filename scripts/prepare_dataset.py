from __future__ import annotations

import argparse
import csv
from pathlib import Path


CONDITIONS = ["fog", "night", "rain", "snow"]
SPLITS = ["train", "val", "test"]

NORMAL_REF_SPLIT_MAP = {
    "train": "train_ref",
    "val": "val_ref",
    "test": "test_ref",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ACDC split CSV files without modifying the dataset."
    )
    parser.add_argument(
        "--raw-data-parent",
        type=Path,
        default=Path("."),
        help="Parent directory that contains data/raw. Default: current project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for split CSV files. Default: data/splits.",
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="Do not create data/splits/normal.csv.",
    )
    return parser.parse_args()


def format_path(path: Path, project_root: Path) -> str:
    path = path.resolve()

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def find_adverse_image_files(image_dir: Path) -> list[Path]:
    return sorted(image_dir.rglob("*_rgb_anon.png"))


def find_normal_image_files(image_dir: Path) -> list[Path]:
    return sorted(image_dir.rglob("*_rgb_ref_anon.png"))


def find_normal_label_files(label_dir: Path) -> list[Path]:
    return sorted(label_dir.rglob("*_gt_ref_labelTrainIds.png"))


def normalize_ref_stem(path: Path) -> str:
    """Normalize normal RGB/GT ref filename stem for matching.

    Examples:
        GOPR0475_frame_000041_rgb_ref_anon.png
        GOPR0475_frame_000041_gt_ref_labelTrainIds.png

    both become:
        GOPR0475_frame_000041
    """
    stem = path.stem

    suffixes = [
        "_rgb_ref_anon",
        "_gt_ref_labelTrainIds",
        "_gt_ref_labelIds",
        "_gt_ref_labelColor",
    ]

    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    return stem


def adverse_image_to_label_path(image_path: Path, raw_dir: Path) -> Path:
    """Convert adverse RGB image path to adverse labelTrainIds path."""
    relative_path = image_path.relative_to(raw_dir / "rgb_anon")
    relative_str = str(relative_path)

    if not relative_str.endswith("_rgb_anon.png"):
        raise ValueError(f"Unexpected adverse image filename: {image_path}")

    label_relative_str = relative_str.replace(
        "_rgb_anon.png",
        "_gt_labelTrainIds.png",
    )
    return raw_dir / "gt" / Path(label_relative_str)


def normal_image_to_label_path(image_path: Path, raw_dir: Path) -> Path:
    """Convert normal reference RGB path to normal reference labelTrainIds path."""
    relative_path = image_path.relative_to(raw_dir / "rgb_anon")
    relative_str = str(relative_path)

    if not relative_str.endswith("_rgb_ref_anon.png"):
        raise ValueError(f"Unexpected normal image filename: {image_path}")

    label_relative_str = relative_str.replace(
        "_rgb_ref_anon.png",
        "_gt_ref_labelTrainIds.png",
    )
    return raw_dir / "gt" / Path(label_relative_str)


def collect_adverse_rows(
    raw_dir: Path,
    split: str,
    project_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for condition in CONDITIONS:
        image_dir = raw_dir / "rgb_anon" / condition / split

        if not image_dir.exists():
            print(f"[WARN] Missing image directory: {image_dir}")
            continue

        image_paths = find_adverse_image_files(image_dir)
        print(f"[INFO] adverse {condition}/{split}: {len(image_paths)} images")

        for image_path in image_paths:
            if split == "test":
                label_path_str = ""
            else:
                label_path = adverse_image_to_label_path(image_path, raw_dir)

                if not label_path.exists():
                    print(f"[WARN] Missing label file: {label_path}")
                    continue

                label_path_str = format_path(label_path, project_root)

            rows.append(
                {
                    "image_path": format_path(image_path, project_root),
                    "label_path": label_path_str,
                    "condition": condition,
                    "split": split,
                }
            )

    return rows


def build_ref_label_index(label_paths: list[Path]) -> dict[str, Path]:
    label_index: dict[str, Path] = {}

    for label_path in label_paths:
        label_index[normalize_ref_stem(label_path)] = label_path

    return label_index


def match_normal_labels(
    image_paths: list[Path],
    label_paths: list[Path],
    raw_dir: Path,
) -> list[Path | None]:
    """Match normal ref RGB images to normal ref GT labels.

    Matching priority:
      1. Direct path replacement
      2. Normalized filename stem
      3. Sorted order, only when counts match
    """
    label_set = set(label_paths)
    label_index = build_ref_label_index(label_paths)
    matched: list[Path | None] = []

    direct_count = 0
    stem_count = 0
    order_count = 0

    # First two strategies.
    for image_path in image_paths:
        label_path = normal_image_to_label_path(image_path, raw_dir)

        if label_path in label_set or label_path.exists():
            matched.append(label_path)
            direct_count += 1
            continue

        key = normalize_ref_stem(image_path)
        label_path = label_index.get(key)

        if label_path is not None:
            matched.append(label_path)
            stem_count += 1
            continue

        matched.append(None)

    # Fallback to order matching if direct/stem failed for all or part of the sequence.
    if any(label_path is None for label_path in matched) and len(image_paths) == len(label_paths):
        matched = [
            current if current is not None else order_label
            for current, order_label in zip(matched, label_paths)
        ]
        order_count = sum(1 for current in matched if current is not None) - direct_count - stem_count

    print(
        f"direct={direct_count}, stem={stem_count}, order={max(order_count, 0)}, "
        f"missing={sum(1 for x in matched if x is None)}"
    )

    return matched


def collect_normal_rows(
    raw_dir: Path,
    project_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for target_split, ref_split in NORMAL_REF_SPLIT_MAP.items():
        for condition in CONDITIONS:
            image_root = raw_dir / "rgb_anon" / condition / ref_split
            label_root = raw_dir / "gt" / condition / ref_split

            if not image_root.exists():
                print(f"[WARN] Missing normal image directory: {image_root}")
                continue

            image_sequence_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])

            if not image_sequence_dirs:
                image_sequence_dirs = [image_root]

            total_images = 0
            total_labels = 0
            total_missing = 0

            for image_sequence_dir in image_sequence_dirs:
                sequence_name = image_sequence_dir.name
                label_sequence_dir = label_root / sequence_name

                image_paths = find_normal_image_files(image_sequence_dir)
                label_paths = find_normal_label_files(label_sequence_dir) if label_sequence_dir.exists() else []

                if len(image_paths) == 0:
                    continue

                print(
                    f"[INFO] normal {condition}/{ref_split}/{sequence_name}: "
                    f"images={len(image_paths)}, labels={len(label_paths)}, match: ",
                    end="",
                )

                matched_labels = match_normal_labels(
                    image_paths=image_paths,
                    label_paths=label_paths,
                    raw_dir=raw_dir,
                )

                for image_path, label_path in zip(image_paths, matched_labels):
                    if label_path is None:
                        label_path_str = ""
                        total_missing += 1
                    else:
                        label_path_str = format_path(label_path, project_root)
                        total_labels += 1

                    rows.append(
                        {
                            "image_path": format_path(image_path, project_root),
                            "label_path": label_path_str,
                            "condition": "normal",
                            "split": target_split,
                        }
                    )

                total_images += len(image_paths)

            print(
                f"[INFO] normal {condition}/{ref_split} -> split={target_split}: "
                f"{total_images} images, matched labels={total_labels}, "
                f"missing labels={total_missing}"
            )

    return rows


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "label_path", "condition", "split"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_split_counts(rows: list[dict[str, str]], name: str) -> None:
    split_counts = {split: 0 for split in SPLITS}
    label_counts = {split: 0 for split in SPLITS}

    for row in rows:
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

        if row["label_path"].strip():
            label_counts[split] = label_counts.get(split, 0) + 1

    for split in SPLITS:
        print(
            f"[INFO] {name} split={split}: "
            f"{split_counts.get(split, 0)} rows, "
            f"{label_counts.get(split, 0)} with labels"
        )


def main() -> None:
    args = parse_args()

    project_root = Path(".").resolve()
    raw_data_parent = args.raw_data_parent.resolve()
    raw_dir = raw_data_parent / "data" / "raw"

    output_dir = args.output_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    if not (raw_dir / "rgb_anon").exists():
        raise FileNotFoundError(f"rgb_anon directory not found: {raw_dir / 'rgb_anon'}")

    if not (raw_dir / "gt").exists():
        print(f"[WARN] gt directory not found: {raw_dir / 'gt'}")

    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] raw_data_parent: {raw_data_parent}")
    print(f"[INFO] raw_dir: {raw_dir}")
    print(f"[INFO] output_dir: {output_dir}")

    for split in SPLITS:
        rows = collect_adverse_rows(
            raw_dir=raw_dir,
            split=split,
            project_root=project_root,
        )

        output_path = output_dir / f"{split}.csv"
        save_csv(rows, output_path)
        print(f"[INFO] Saved {output_path} ({len(rows)} rows)")

    if not args.skip_normal:
        normal_rows = collect_normal_rows(
            raw_dir=raw_dir,
            project_root=project_root,
        )

        normal_output_path = output_dir / "normal.csv"
        save_csv(normal_rows, normal_output_path)
        print(f"[INFO] Saved {normal_output_path} ({len(normal_rows)} rows)")
        print_split_counts(normal_rows, "normal.csv")

        if len(normal_rows) == 0:
            print("[WARN] normal.csv has 0 rows. Check rgb_anon/*/*_ref directories.")

    print("[DONE] Created split CSV files.")
    print("[DONE] Original dataset folders were not modified.")


if __name__ == "__main__":
    main()
