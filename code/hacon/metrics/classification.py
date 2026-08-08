from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClassificationMetrics:
    auc: Tensor
    accuracy: Tensor
    balanced_accuracy: Tensor
    sensitivity: Tensor
    specificity: Tensor
    precision: Tensor
    f1: Tensor


def binary_confusion(
    scores: Tensor,
    labels: Tensor,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    prediction = scores >= threshold
    target = labels.bool()
    true_positive = (prediction & target).sum().float()
    true_negative = (~prediction & ~target).sum().float()
    false_positive = (prediction & ~target).sum().float()
    false_negative = (~prediction & target).sum().float()
    return true_positive, true_negative, false_positive, false_negative


def binary_auc(scores: Tensor, labels: Tensor) -> Tensor:
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    positives = sorted_labels.sum()
    negatives = sorted_labels.numel() - positives
    if positives == 0 or negatives == 0:
        return torch.tensor(float("nan"), device=scores.device)
    true_positive = torch.cumsum(sorted_labels, dim=0)
    false_positive = torch.cumsum(1.0 - sorted_labels, dim=0)
    true_positive_rate = torch.cat((torch.zeros(1, device=scores.device), true_positive / positives))
    false_positive_rate = torch.cat((torch.zeros(1, device=scores.device), false_positive / negatives))
    return torch.trapz(true_positive_rate, false_positive_rate)


def binary_classification_metrics(
    scores: Tensor,
    labels: Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> ClassificationMetrics:
    tp, tn, fp, fn = binary_confusion(scores, labels, threshold)
    sensitivity = tp / (tp + fn).clamp_min(epsilon)
    specificity = tn / (tn + fp).clamp_min(epsilon)
    precision = tp / (tp + fp).clamp_min(epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn).clamp_min(epsilon)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity).clamp_min(epsilon)
    return ClassificationMetrics(
        auc=binary_auc(scores, labels),
        accuracy=accuracy,
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
    )


def one_vs_rest_auc(probabilities: Tensor, labels: Tensor, classes: int) -> Tensor:
    scores = [
        binary_auc(probabilities[:, index], labels == index)
        for index in range(classes)
    ]
    return torch.stack(scores)


def macro_auc(probabilities: Tensor, labels: Tensor, classes: int) -> Tensor:
    return torch.nanmean(one_vs_rest_auc(probabilities, labels, classes))


def topk_accuracy(logits: Tensor, labels: Tensor, k: int = 1) -> Tensor:
    indices = logits.topk(k, dim=1).indices
    return (indices == labels[:, None]).any(dim=1).float().mean()


def brier_score(probabilities: Tensor, labels: Tensor) -> Tensor:
    return (probabilities - labels.float()).square().mean()


def negative_log_likelihood(probabilities: Tensor, labels: Tensor, epsilon: float = 1e-7) -> Tensor:
    selected = torch.where(labels.bool(), probabilities, 1.0 - probabilities)
    return -torch.log(selected.clamp_min(epsilon)).mean()


def expected_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    bins: int = 15,
) -> Tensor:
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    error = torch.zeros((), device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (probabilities > lower) & (probabilities <= upper)
        if selected.any():
            confidence = probabilities[selected].mean()
            accuracy = labels[selected].float().mean()
            error = error + selected.float().mean() * torch.abs(confidence - accuracy)
    return error


def maximum_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    bins: int = 15,
) -> Tensor:
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    errors: list[Tensor] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (probabilities > lower) & (probabilities <= upper)
        if selected.any():
            errors.append(torch.abs(probabilities[selected].mean() - labels[selected].float().mean()))
    if not errors:
        return torch.zeros((), device=probabilities.device)
    return torch.stack(errors).max()

