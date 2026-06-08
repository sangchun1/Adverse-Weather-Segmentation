from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset

from awseg.transforms import build_transform


ACDC_CITYSCAPES_CLASSES = [
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


ACDC_CONDITIONS = {"fog", "night", "rain", "snow"}


class ACDCSegmentationDataset(Dataset):
    """ACDC semantic segmentation dataset using CSV split files.

    Expected CSV columns:
        - image_path
        - label_path
        - condition
        - split

    Required CSV column:
        - image_path

    Notes:
        - label_path can be empty for test samples.
        - condition is optional. If missing, it is inferred from image_path.
        - transform is called as transform(image, mask, condition=condition).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transform: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.split = str(split).lower()

        data_config = config["data"]
        self.root = Path(data_config.get("root", "."))
        self.split_dir = self._resolve_dir(data_config.get("split_dir", "data/splits"))
        self.csv_path = self.split_dir / f"{self.split}.csv"

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.csv_path}. "
                "Run `python scripts/prepare_dataset.py` first."
            )

        self.samples = self._load_csv(self.csv_path)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in split file: {self.csv_path}")

        self.transform = transform if transform is not None else build_transform(
            config,
            self.split,
        )

    def _resolve_dir(self, path: str | Path) -> Path:
        """Resolve a directory path.

        If path is absolute, return it as-is.
        Otherwise, interpret it relative to data.root.
        """
        path = Path(path)

        if path.is_absolute():
            return path

        return self.root / path

    def _resolve_file(self, path: str | Path) -> Path:
        """Resolve a file path.

        The split CSV may contain either absolute paths or paths relative to
        data.root / project root. This method supports both cases.
        """
        path = Path(path)

        if path.is_absolute():
            return path

        root_relative = self.root / path
        if root_relative.exists():
            return root_relative

        # Fallback for cases where CSV paths are already relative to CWD.
        return path

    @staticmethod
    def _is_missing_path(path: Any) -> bool:
        if path is None:
            return True

        path_str = str(path).strip()
        return path_str == "" or path_str.lower() in {"none", "nan", "null"}

    @staticmethod
    def _normalize_condition(condition: Any) -> str:
        if condition is None:
            return "unknown"

        condition = str(condition).strip().lower()
        if condition in {"", "none", "nan", "null"}:
            return "unknown"

        return condition

    @classmethod
    def _infer_condition_from_path(cls, image_path: str) -> str:
        """Infer weather condition from path parts.

        Example:
            data/raw/rgb_anon/fog/train/...
            -> fog
        """
        path_parts = [part.lower() for part in Path(image_path).parts]

        for condition in ACDC_CONDITIONS:
            if condition in path_parts:
                return condition

        # Fallback for paths that contain condition name inside filename.
        lower_path = image_path.lower()
        for condition in ACDC_CONDITIONS:
            if f"/{condition}/" in lower_path or f"\\{condition}\\" in lower_path:
                return condition

        return "unknown"

    def _load_csv(self, csv_path: Path) -> list[Dict[str, str]]:
        samples: list[Dict[str, str]] = []

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            required_columns = {"image_path"}
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {csv_path}: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                image_path = row.get("image_path", "")
                label_path = row.get("label_path", "")
                row_split = row.get("split", self.split)

                if self._is_missing_path(image_path):
                    continue

                condition = self._normalize_condition(row.get("condition"))
                if condition == "unknown":
                    condition = self._infer_condition_from_path(image_path)

                samples.append(
                    {
                        "image_path": image_path,
                        "label_path": label_path,
                        "condition": condition,
                        "split": row_split,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        image_path = self._resolve_file(sample["image_path"])
        label_path_raw = sample.get("label_path", "")
        condition = self._normalize_condition(sample.get("condition"))

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()

        mask = None
        label_path = None
        has_label = not self._is_missing_path(label_path_raw)

        if has_label:
            label_path = self._resolve_file(label_path_raw)

            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")

            # Keep mask as single-channel class ID image.
            # copy() closes the underlying file handle safely.
            with Image.open(label_path) as m:
                mask = m.copy()
        elif self.split in {"train", "val"}:
            raise ValueError(
                f"Label path is missing for {self.split} sample: {image_path}"
            )

        # Pass condition metadata to transforms so enhancement can be applied
        # selectively, e.g. enhancement.apply_conditions: ["night"].
        try:
            image, mask = self.transform(image, mask, condition=condition)
        except TypeError:
            # Backward compatibility for old transforms that accept only image/mask.
            image, mask = self.transform(image, mask)

        output: Dict[str, Any] = {
            "image": image,
            "condition": condition,
            "image_path": str(image_path),
            "split": sample.get("split", self.split),
            "index": index,
        }

        if mask is not None:
            output["mask"] = mask

        if label_path is not None:
            output["label_path"] = str(label_path)

        return output


def build_dataset(config: Dict[str, Any], split: str) -> ACDCSegmentationDataset:
    """Build ACDC dataset from config."""
    return ACDCSegmentationDataset(config=config, split=split)


def get_class_names() -> list[str]:
    """Return 19 ACDC/Cityscapes semantic class names."""
    return ACDC_CITYSCAPES_CLASSES.copy()
