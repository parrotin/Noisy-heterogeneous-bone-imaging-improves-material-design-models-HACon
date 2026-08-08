from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: Tensor
    sensitivity: Tensor
    specificity: Tensor
    precision: Tensor
    jaccard: Tensor
    volume_similarity: Tensor


def _flatten(value: Tensor) -> Tensor:
    return value.reshape(value.shape[0], -1)


def confusion_counts(
    prediction: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    predicted = _flatten(prediction >= threshold)
    expected = _flatten(target.bool())
    true_positive = (predicted & expected).sum(dim=1).float()
    true_negative = (~predicted & ~expected).sum(dim=1).float()
    false_positive = (predicted & ~expected).sum(dim=1).float()
    false_negative = (~predicted & expected).sum(dim=1).float()
    return true_positive, true_negative, false_positive, false_negative


def dice_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    predicted = _flatten(prediction.float())
    expected = _flatten(target.float())
    intersection = (predicted * expected).sum(dim=1)
    denominator = predicted.sum(dim=1) + expected.sum(dim=1)
    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def jaccard_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    predicted = _flatten(prediction.float())
    expected = _flatten(target.float())
    intersection = (predicted * expected).sum(dim=1)
    union = predicted.sum(dim=1) + expected.sum(dim=1) - intersection
    return (intersection + epsilon) / (union + epsilon)


def sensitivity_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    true_positive, _, _, false_negative = confusion_counts(prediction, target)
    return (true_positive + epsilon) / (true_positive + false_negative + epsilon)


def specificity_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    _, true_negative, false_positive, _ = confusion_counts(prediction, target)
    return (true_negative + epsilon) / (true_negative + false_positive + epsilon)


def precision_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    true_positive, _, false_positive, _ = confusion_counts(prediction, target)
    return (true_positive + epsilon) / (true_positive + false_positive + epsilon)


def volume_similarity(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    predicted = _flatten(prediction.float()).sum(dim=1)
    expected = _flatten(target.float()).sum(dim=1)
    return 1.0 - torch.abs(predicted - expected) / (predicted + expected + epsilon)


def binary_metrics(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> SegmentationMetrics:
    binary = prediction >= threshold
    return SegmentationMetrics(
        dice=dice_score(binary, target),
        sensitivity=sensitivity_score(binary, target),
        specificity=specificity_score(binary, target),
        precision=precision_score(binary, target),
        jaccard=jaccard_score(binary, target),
        volume_similarity=volume_similarity(binary, target),
    )


def multiclass_dice(
    prediction: Tensor,
    target: Tensor,
    classes: int,
    include_background: bool = False,
) -> Tensor:
    labels = prediction.argmax(dim=1)
    start = 0 if include_background else 1
    scores = [
        dice_score((labels == label).float(), (target == label).float())
        for label in range(start, classes)
    ]
    return torch.stack(scores, dim=1)


def surface_points(mask: Tensor, spacing: tuple[float, float, float]) -> Tensor:
    value = mask.float().unsqueeze(0) if mask.ndim == 3 else mask.float()
    eroded = -torch.nn.functional.max_pool3d(-value, 3, stride=1, padding=1)
    surface = (value > 0.5) & (eroded <= 0.5)
    points = torch.nonzero(surface.squeeze(), as_tuple=False).float()
    scale = torch.tensor(spacing, device=points.device, dtype=points.dtype)
    return points * scale


def surface_distances(
    prediction: Tensor,
    target: Tensor,
    spacing: tuple[float, float, float],
) -> Tensor:
    first = surface_points(prediction, spacing)
    second = surface_points(target, spacing)
    if first.numel() == 0 or second.numel() == 0:
        return torch.tensor([float("inf")], device=prediction.device)
    distances = torch.cdist(first, second)
    return torch.cat((distances.min(dim=1).values, distances.min(dim=0).values))


def hausdorff_95(
    prediction: Tensor,
    target: Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    distances = surface_distances(prediction, target, spacing)
    return torch.quantile(distances, 0.95)


def average_surface_distance(
    prediction: Tensor,
    target: Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    return surface_distances(prediction, target, spacing).mean()


def normalized_surface_dice(
    prediction: Tensor,
    target: Tensor,
    tolerance: float,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    distances = surface_distances(prediction, target, spacing)
    return (distances <= tolerance).float().mean()


class RunningSegmentationMetrics:
    def __init__(self) -> None:
        self.values: list[SegmentationMetrics] = []

    def update(self, prediction: Tensor, target: Tensor, threshold: float = 0.5) -> None:
        self.values.append(binary_metrics(prediction.detach(), target.detach(), threshold))

    def compute(self) -> SegmentationMetrics:
        if not self.values:
            raise ValueError("no metric observations")
        return SegmentationMetrics(
            dice=torch.cat([value.dice for value in self.values]).mean(),
            sensitivity=torch.cat([value.sensitivity for value in self.values]).mean(),
            specificity=torch.cat([value.specificity for value in self.values]).mean(),
            precision=torch.cat([value.precision for value in self.values]).mean(),
            jaccard=torch.cat([value.jaccard for value in self.values]).mean(),
            volume_similarity=torch.cat([value.volume_similarity for value in self.values]).mean(),
        )

    def reset(self) -> None:
        self.values.clear()

