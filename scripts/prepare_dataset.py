from __future__ import annotations

import argparse
import csv
from pathlib import Path


CONDITIONS = ["fog", "night", "rain", "snow"]
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ACDC split CSV files without modifying the dataset."
    )

    parser.add_argument(
        "--raw-data-parent",
        type=Path,
        default=Path("."),
        help=(
            "Parent directory that contains data/raw. "
            "Default: current project root."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help=(
            "Output directory for split CSV files. "
            "Default: data/split under the current project."
        ),
    )

    return parser.parse_args()


def format_path(path: Path, project_root: Path) -> str:
    """
    If the file is inside the current project, store a relative path.
    If the file is outside the current project, store an absolute path.
    """
    path = path.resolve()

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def get_label_path(image_path: Path, raw_dir: Path) -> Path:
    """
    Convert image path to label path.

    Example:
    /.../data/raw/rgb_anon/fog/train/seq/file_rgb_anon.png

    becomes:
    /.../data/raw/gt/fog/train/seq/file_gt_labelTrainIds.png
    """
    relative_path = image_path.relative_to(raw_dir / "rgb_anon")

    label_relative_path = Path(
        str(relative_path).replace("_rgb_anon.png", "_gt_labelTrainIds.png")
    )

    return raw_dir / "gt" / label_relative_path


def collect_rows(
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

        image_paths = sorted(image_dir.rglob("*_rgb_anon.png"))

        for image_path in image_paths:
            if split == "test":
                label_path_str = ""
            else:
                label_path = get_label_path(image_path, raw_dir)

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


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "label_path", "condition", "split"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    project_root = Path(".").resolve()

    raw_data_parent = args.raw_data_parent.resolve()
    raw_dir = raw_data_parent / "data" / "raw"

    # 중요:
    # output_dir은 raw_data_parent 기준이 아니라,
    # 현재 프로젝트 기준으로 유지됨.
    output_dir = args.output_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    if not (raw_dir / "rgb_anon").exists():
        raise FileNotFoundError(
            f"rgb_anon directory not found: {raw_dir / 'rgb_anon'}"
        )

    print(f"[INFO] project_root:     {project_root}")
    print(f"[INFO] raw_data_parent: {raw_data_parent}")
    print(f"[INFO] raw_dir:         {raw_dir}")
    print(f"[INFO] output_dir:      {output_dir}")

    for split in SPLITS:
        rows = collect_rows(
            raw_dir=raw_dir,
            split=split,
            project_root=project_root,
        )

        output_path = output_dir / f"{split}.csv"
        save_csv(rows, output_path)

        print(f"[INFO] Saved {output_path} ({len(rows)} rows)")

    print("[DONE] Created split CSV files.")
    print("[DONE] Original dataset folders were not modified.")


if __name__ == "__main__":
    main()