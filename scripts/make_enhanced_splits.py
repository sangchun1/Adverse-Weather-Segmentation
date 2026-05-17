from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create split CSVs for offline enhanced images."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["sci", "zero_dce", "retinexformer"],
        help="Enhanced image method name.",
    )

    parser.add_argument(
        "--condition",
        type=str,
        default="night",
        help="ACDC condition. Default: night",
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Original split CSV directory.",
    )

    parser.add_argument(
        "--output-split-dir",
        type=Path,
        default=None,
        help="Output split CSV directory. Default: data/splits/enhanced/<method>",
    )

    parser.add_argument(
        "--enhanced-data-parent",
        type=Path,
        default=Path("/home/user/DATA/awseg"),
        help="Parent directory that contains data/enhanced.",
    )

    parser.add_argument(
        "--keep-absolute",
        action="store_true",
        help="Write enhanced image paths as absolute paths.",
    )

    return parser.parse_args()


def normalize_method(method: str) -> str:
    return method.lower().replace("-", "_")


def extract_relative_image_path(image_path: str, condition: str) -> Path:
    """기존 image_path에서 rgb_anon/<condition>/ 뒤의 상대경로만 추출한다.

    예:
    /home/user/DATA/awseg/data/raw/rgb_anon/night/train/xxx.png
    -> train/xxx.png

    data/raw/rgb_anon/night/val/yyy.png
    -> val/yyy.png
    """

    marker = f"rgb_anon/{condition}/"
    normalized = image_path.replace("\\", "/")

    if marker not in normalized:
        raise ValueError(
            f"Could not find marker '{marker}' in image_path: {image_path}"
        )

    relative = normalized.split(marker, maxsplit=1)[1]
    return Path(relative)


def build_enhanced_image_path(
    original_image_path: str,
    method: str,
    condition: str,
    enhanced_data_parent: Path,
    keep_absolute: bool,
) -> str:
    relative_image_path = extract_relative_image_path(
        image_path=original_image_path,
        condition=condition,
    )

    enhanced_path = (
        enhanced_data_parent
        / "data"
        / "enhanced"
        / method
        / "rgb_anon"
        / condition
        / relative_image_path
    )

    if keep_absolute:
        return str(enhanced_path.resolve())

    try:
        return str(enhanced_path.resolve().relative_to(Path(".").resolve()))
    except ValueError:
        return str(enhanced_path.resolve())


def convert_split_csv(
    input_csv: Path,
    output_csv: Path,
    method: str,
    condition: str,
    enhanced_data_parent: Path,
    keep_absolute: bool,
) -> None:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")

        if "image_path" not in fieldnames:
            raise ValueError(f"'image_path' column not found in {input_csv}")

        rows = []

        for row in reader:
            # condition column이 있으면 night만 유지.
            # 이미 night 전용 split이면 모든 row가 유지됨.
            if "condition" in row and row["condition"] != condition:
                continue

            row = dict(row)
            row["image_path"] = build_enhanced_image_path(
                original_image_path=row["image_path"],
                method=method,
                condition=condition,
                enhanced_data_parent=enhanced_data_parent,
                keep_absolute=keep_absolute,
            )

            # label_path는 건드리지 않는다.
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved {output_csv} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    method = normalize_method(args.method)

    if args.output_split_dir is None:
        output_split_dir = Path("data") / "splits" / "enhanced" / method
    else:
        output_split_dir = args.output_split_dir

    split_files = ["train.csv", "val.csv", "test.csv"]

    print(f"[INFO] method: {method}")
    print(f"[INFO] condition: {args.condition}")
    print(f"[INFO] split_dir: {args.split_dir}")
    print(f"[INFO] output_split_dir: {output_split_dir}")
    print(f"[INFO] enhanced_data_parent: {args.enhanced_data_parent}")

    for split_file in split_files:
        input_csv = args.split_dir / split_file
        output_csv = output_split_dir / split_file

        if not input_csv.exists():
            print(f"[WARN] Missing split file: {input_csv}")
            continue

        convert_split_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            method=method,
            condition=args.condition,
            enhanced_data_parent=args.enhanced_data_parent,
            keep_absolute=args.keep_absolute,
        )

    print("[DONE] Enhanced split CSVs created.")
    print("[DONE] Split membership was preserved. Images were not re-split.")


if __name__ == "__main__":
    main()