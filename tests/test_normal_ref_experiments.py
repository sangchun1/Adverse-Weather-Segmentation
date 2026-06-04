from __future__ import annotations

import unittest
import sys
import types
from unittest.mock import patch


class FakeSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)


torch_module = types.ModuleType("torch")
torch_module.Tensor = object
torch_module.no_grad = lambda: (lambda fn: fn)
torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_module.nn = types.SimpleNamespace(Module=object)
torch_module.optim = types.SimpleNamespace(
    Optimizer=object,
    AdamW=object,
    Adam=object,
    SGD=object,
    lr_scheduler=types.SimpleNamespace(LRScheduler=object, CosineAnnealingLR=object),
)
utils_module = types.ModuleType("torch.utils")
data_module = types.ModuleType("torch.utils.data")
data_module.ConcatDataset = object
data_module.DataLoader = object
data_module.Dataset = object
data_module.Subset = FakeSubset
utils_module.data = data_module
torch_module.utils = utils_module

sys.modules.setdefault("torch", torch_module)
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("torch.utils", utils_module)
sys.modules.setdefault("torch.utils.data", data_module)
sys.modules.setdefault(
    "awseg.losses",
    types.SimpleNamespace(build_loss=lambda config: None),
)
sys.modules.setdefault(
    "awseg.logger",
    types.SimpleNamespace(build_logger=lambda config: None),
)
sys.modules.setdefault(
    "awseg.metrics",
    types.SimpleNamespace(
        SegmentationMetric=object,
        format_class_iou=lambda class_iou, class_names: "",
    ),
)
sys.modules.setdefault(
    "awseg.models",
    types.SimpleNamespace(build_model=lambda config: None),
)
sys.modules.setdefault(
    "awseg.utils",
    types.SimpleNamespace(
        AverageMeter=object,
        count_parameters=lambda model: 0,
        ensure_dir=lambda path: path,
        format_metrics=lambda metrics, prefix="": "",
        get_device=lambda: "cpu",
        get_lr=lambda optimizer: 0.0,
        load_config=lambda path: {},
        save_checkpoint=lambda state, path: None,
        save_config=lambda config, path: None,
        set_seed=lambda seed: None,
    ),
)

from awseg import evaluate, train


class DummyDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class NormalRefExperimentTest(unittest.TestCase):
    def test_train_normal_filter_uses_split_label_and_condition_path(self) -> None:
        dataset = DummyDataset(
            [
                {
                    "image_path": "data/raw/rgb_anon/snow/train_ref/a.png",
                    "label_path": "data/raw/gt/snow/train_ref/a.png",
                    "split": "train",
                },
                {
                    "image_path": "data/raw/rgb_anon/snow/val_ref/b.png",
                    "label_path": "data/raw/gt/snow/val_ref/b.png",
                    "split": "val",
                },
                {
                    "image_path": "data/raw/rgb_anon/rain/train_ref/c.png",
                    "label_path": "data/raw/gt/rain/train_ref/c.png",
                    "split": "train",
                },
                {
                    "image_path": "data/raw/rgb_anon/snow/train_ref/d.png",
                    "label_path": "",
                    "split": "train",
                },
            ]
        )

        subset = train.filter_dataset_by_split_column(
            dataset,
            target_split="train",
            require_label=True,
            condition="snow",
        )

        self.assertEqual(subset.indices, [0])

    def test_train_adverse_source_keeps_existing_condition_column_filter(self) -> None:
        dataset = DummyDataset(
            [
                {"condition": "snow", "split": "train", "label_path": "snow.png"},
                {"condition": "rain", "split": "train", "label_path": "rain.png"},
            ]
        )

        with patch.object(train, "build_dataset", return_value=dataset):
            subset = train.build_training_dataset(
                config={"data": {"root": ".", "split_dir": "data/splits"}},
                split="train",
                condition="snow",
                include_normal=False,
                normal_split="normal",
                train_source="adverse",
            )

        self.assertEqual(subset.indices, [0])

    def test_evaluate_normal_filter_can_include_all_or_one_ref_condition(self) -> None:
        dataset = DummyDataset(
            [
                {
                    "image_path": "data/raw/rgb_anon/snow/val_ref/a.png",
                    "label_path": "data/raw/gt/snow/val_ref/a.png",
                    "split": "val",
                },
                {
                    "image_path": "data/raw/rgb_anon/rain/val_ref/b.png",
                    "label_path": "data/raw/gt/rain/val_ref/b.png",
                    "split": "val",
                },
                {
                    "image_path": "data/raw/rgb_anon/fog/train_ref/c.png",
                    "label_path": "data/raw/gt/fog/train_ref/c.png",
                    "split": "train",
                },
            ]
        )

        all_subset = evaluate.filter_dataset_by_split_column(
            dataset,
            target_split="val",
            require_label=True,
            condition=None,
        )
        snow_subset = evaluate.filter_dataset_by_split_column(
            dataset,
            target_split="val",
            require_label=True,
            condition="snow",
        )

        self.assertEqual(all_subset.indices, [0, 1])
        self.assertEqual(snow_subset.indices, [0])

    def test_normal_summary_uses_ref_condition_label(self) -> None:
        dataset = DummyDataset([{"condition": "normal"}])
        result = {
            "miou": 0.5,
            "class_iou": [0.1, 0.2],
            "condition_results": {
                "normal": {
                    "miou": 0.5,
                    "class_iou": [0.1, 0.2],
                    "num_images": 1,
                }
            },
        }

        summary = evaluate.make_split_summary(
            split="normal",
            condition="snow",
            dataloader=type("Loader", (), {"dataset": dataset})(),
            result=result,
            class_names=["road", "sidewalk"],
            source_csv="normal.csv",
            csv_split_filter="val",
            is_normal_ref=True,
        )

        self.assertEqual(summary["condition"], "snow_ref")
        self.assertIn("snow_ref", summary["condition_results"])
        self.assertNotIn("normal", summary["condition_results"])


if __name__ == "__main__":
    unittest.main()
