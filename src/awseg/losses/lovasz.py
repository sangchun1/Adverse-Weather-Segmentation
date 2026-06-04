from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension with respect to sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)

    jaccard = 1.0 - intersection / union.clamp_min(1e-7)

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def _flatten_probas(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions in [B, C, H, W] and labels in [B, H, W]."""
    if probas.ndim != 4:
        raise ValueError(f"probas must have shape [B, C, H, W], got {tuple(probas.shape)}")
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B, H, W], got {tuple(labels.shape)}")

    if probas.shape[-2:] != labels.shape[-2:]:
        probas = F.interpolate(
            probas,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    c = probas.shape[1]

    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.contiguous().view(-1).long()

    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]

    return probas, labels


def _lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | list[int] = "present",
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss on flattened predictions."""
    if probas.numel() == 0:
        return probas.sum() * 0.0

    c = probas.shape[1]
    losses = []

    if classes in {"all", "present"}:
        class_to_sum = list(range(c))
    else:
        class_to_sum = list(classes)

    for class_idx in class_to_sum:
        fg = (labels == class_idx).float()

        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probas[:, class_idx]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if len(losses) == 0:
        return probas.sum() * 0.0

    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Multiclass Lovasz-Softmax Loss.

    This implementation is self-contained and does not require
    segmentation_models_pytorch.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        classes: str | list[int] = "present",
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.classes = classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(pred, dim=1)
        prob_flat, target_flat = _flatten_probas(
            probas=prob,
            labels=target,
            ignore_index=self.ignore_index,
        )
        return _lovasz_softmax_flat(
            probas=prob_flat,
            labels=target_flat,
            classes=self.classes,
        )
