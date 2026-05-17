from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class SegmentationMetric:
    """Confusion-matrix based semantic segmentation metrics.

    Supports:
        - class-wise IoU
        - mean IoU

    Args:
        num_classes: Number of valid semantic classes.
        ignore_index: Label index to ignore. For ACDC/Cityscapes trainIds,
            this is usually 255.
        device: Device used to store confusion matrix. If None, CPU is used.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=self.device,
        )

    def reset(self) -> None:
        """Reset accumulated confusion matrix."""
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update confusion matrix with a batch.

        Args:
            preds: Either logits with shape [B, C, H, W] or predicted labels
                with shape [B, H, W].
            targets: Ground-truth labels with shape [B, H, W].

        Notes:
            Rows of confusion matrix are ground-truth classes.
            Columns of confusion matrix are predicted classes.
        """
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)

        if preds.shape != targets.shape:
            raise ValueError(
                f"preds and targets must have the same shape after argmax. "
                f"Got preds={tuple(preds.shape)}, targets={tuple(targets.shape)}"
            )

        preds = preds.to(self.device).long()
        targets = targets.to(self.device).long()

        valid_mask = targets != self.ignore_index
        valid_mask &= targets >= 0
        valid_mask &= targets < self.num_classes
        valid_mask &= preds >= 0
        valid_mask &= preds < self.num_classes

        if valid_mask.sum() == 0:
            return

        targets = targets[valid_mask]
        preds = preds[valid_mask]

        indices = targets * self.num_classes + preds
        confmat = torch.bincount(
            indices,
            minlength=self.num_classes * self.num_classes,
        )
        confmat = confmat.reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += confmat

    @torch.no_grad()
    def compute(self) -> Dict[str, Any]:
        """Compute mIoU and class-wise IoU.

        Returns:
            Dictionary containing:
                - miou: mean IoU over classes with valid union.
                - class_iou: list of IoU values for each class.
                - confusion_matrix: accumulated confusion matrix on CPU.
        """
        confmat = self.confusion_matrix.float()

        true_positive = torch.diag(confmat)
        false_positive = confmat.sum(dim=0) - true_positive
        false_negative = confmat.sum(dim=1) - true_positive

        union = true_positive + false_positive + false_negative

        class_iou = torch.full(
            (self.num_classes,),
            float("nan"),
            dtype=torch.float32,
            device=confmat.device,
        )

        valid = union > 0
        class_iou[valid] = true_positive[valid] / union[valid]

        if valid.any():
            miou = torch.nanmean(class_iou).item()
        else:
            miou = 0.0

        return {
            "miou": float(miou),
            "class_iou": class_iou.detach().cpu().tolist(),
            "confusion_matrix": self.confusion_matrix.detach().cpu(),
        }

    @torch.no_grad()
    def get_class_iou(self) -> list[float]:
        """Return only class-wise IoU values."""
        return self.compute()["class_iou"]

    @torch.no_grad()
    def get_miou(self) -> float:
        """Return only mean IoU."""
        return self.compute()["miou"]


def intersection_and_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class intersection and union for one batch.

    This helper is useful for debugging or custom evaluation code.

    Args:
        preds: Predicted labels [B, H, W] or logits [B, C, H, W].
        targets: Ground-truth labels [B, H, W].
        num_classes: Number of classes.
        ignore_index: Label index to ignore.

    Returns:
        intersection: Tensor of shape [num_classes].
        union: Tensor of shape [num_classes].
    """
    if preds.ndim == 4:
        preds = torch.argmax(preds, dim=1)

    if preds.shape != targets.shape:
        raise ValueError(
            f"preds and targets must have the same shape after argmax. "
            f"Got preds={tuple(preds.shape)}, targets={tuple(targets.shape)}"
        )

    preds = preds.long()
    targets = targets.long()

    valid_mask = targets != ignore_index
    valid_mask &= targets >= 0
    valid_mask &= targets < num_classes
    valid_mask &= preds >= 0
    valid_mask &= preds < num_classes

    preds = preds[valid_mask]
    targets = targets[valid_mask]

    intersection = preds[preds == targets]

    area_intersection = torch.bincount(intersection, minlength=num_classes)
    area_pred = torch.bincount(preds, minlength=num_classes)
    area_target = torch.bincount(targets, minlength=num_classes)

    area_union = area_pred + area_target - area_intersection

    return area_intersection.float(), area_union.float()


def compute_miou_from_class_iou(class_iou: list[float]) -> float:
    """Compute mean IoU from a class IoU list while ignoring NaN values."""
    values = torch.tensor(class_iou, dtype=torch.float32)

    valid = ~torch.isnan(values)

    if valid.sum() == 0:
        return 0.0

    return float(values[valid].mean().item())


def format_class_iou(
    class_iou: list[float],
    class_names: Optional[list[str]] = None,
    digits: int = 4,
) -> str:
    """Format class-wise IoU values for console output.

    Args:
        class_iou: List of class IoU values.
        class_names: Optional class names.
        digits: Number of decimal places.

    Returns:
        Multiline string.
    """
    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(len(class_iou))]

    lines = []

    for name, iou in zip(class_names, class_iou):
        if iou != iou:  # NaN check
            iou_text = "nan"
        else:
            iou_text = f"{iou:.{digits}f}"

        lines.append(f"{name}: {iou_text}")

    return "\n".join(lines)
