from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EnsemblePrediction:
    mean: Tensor
    variance: Tensor
    standard_deviation: Tensor
    lower: Tensor
    upper: Tensor
    members: Tensor


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float
    gap: float


def ensemble_prediction(
    members: Tensor,
    confidence: float = 0.90,
) -> EnsemblePrediction:
    if members.ndim < 2:
        raise ValueError("member predictions require ensemble and sample dimensions")
    mean = members.mean(dim=0)
    variance = members.var(dim=0, unbiased=False)
    standard_deviation = variance.sqrt()
    alpha = (1.0 - confidence) / 2.0
    lower = torch.quantile(members, alpha, dim=0)
    upper = torch.quantile(members, 1.0 - alpha, dim=0)
    return EnsemblePrediction(mean, variance, standard_deviation, lower, upper, members)


def prediction_interval_coverage(
    target: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    return ((target >= lower) & (target <= upper)).float().mean()


def prediction_interval_width(lower: Tensor, upper: Tensor) -> Tensor:
    return (upper - lower).mean()


def uncertainty_error_correlation(
    prediction: Tensor,
    target: Tensor,
    uncertainty: Tensor,
) -> Tensor:
    error = torch.abs(prediction - target).flatten()
    uncertainty_flat = uncertainty.flatten()
    error_centered = error - error.mean()
    uncertainty_centered = uncertainty_flat - uncertainty_flat.mean()
    numerator = (error_centered * uncertainty_centered).sum()
    denominator = error_centered.square().sum().sqrt() * uncertainty_centered.square().sum().sqrt()
    return numerator / denominator.clamp_min(1e-7)


def ood_uncertainty_ratio(id_uncertainty: Tensor, ood_uncertainty: Tensor) -> Tensor:
    return ood_uncertainty.mean() / id_uncertainty.mean().clamp_min(1e-7)


def predictive_entropy(probabilities: Tensor, epsilon: float = 1e-7) -> Tensor:
    values = probabilities.clamp(epsilon, 1.0 - epsilon)
    return -(values * values.log() + (1.0 - values) * (1.0 - values).log())


def multiclass_entropy(probabilities: Tensor, epsilon: float = 1e-7) -> Tensor:
    values = probabilities.clamp_min(epsilon)
    return -(values * values.log()).sum(dim=-1)


def mutual_information(member_probabilities: Tensor, epsilon: float = 1e-7) -> Tensor:
    mean_probability = member_probabilities.mean(dim=0)
    total_entropy = multiclass_entropy(mean_probability, epsilon)
    expected_entropy = multiclass_entropy(member_probabilities, epsilon).mean(dim=0)
    return total_entropy - expected_entropy


def variation_ratio(member_labels: Tensor) -> Tensor:
    count = member_labels.shape[0]
    modes = torch.mode(member_labels, dim=0).values
    agreements = (member_labels == modes).sum(dim=0)
    return 1.0 - agreements.float() / count


def calibration_bins(
    probabilities: Tensor,
    labels: Tensor,
    bins: int = 15,
) -> list[CalibrationBin]:
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    result: list[CalibrationBin] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (probabilities > lower) & (probabilities <= upper)
        count = int(selected.sum().item())
        if count == 0:
            result.append(CalibrationBin(float(lower), float(upper), 0, 0.0, 0.0, 0.0))
            continue
        confidence = float(probabilities[selected].mean().item())
        accuracy = float(labels[selected].float().mean().item())
        result.append(
            CalibrationBin(
                float(lower),
                float(upper),
                count,
                confidence,
                accuracy,
                abs(confidence - accuracy),
            )
        )
    return result


def expected_calibration_error_from_bins(bins: list[CalibrationBin]) -> float:
    total = sum(item.count for item in bins)
    if total == 0:
        return 0.0
    return sum(item.count / total * item.gap for item in bins)


def risk_coverage_curve(
    errors: Tensor,
    uncertainty: Tensor,
) -> tuple[Tensor, Tensor]:
    order = torch.argsort(uncertainty)
    sorted_errors = errors.flatten()[order]
    cumulative_risk = torch.cumsum(sorted_errors, dim=0) / torch.arange(
        1,
        sorted_errors.numel() + 1,
        device=errors.device,
    )
    coverage = torch.arange(
        1,
        sorted_errors.numel() + 1,
        device=errors.device,
    ).float() / sorted_errors.numel()
    return coverage, cumulative_risk


def area_under_risk_coverage(errors: Tensor, uncertainty: Tensor) -> Tensor:
    coverage, risk = risk_coverage_curve(errors, uncertainty)
    return torch.trapz(risk, coverage)


def uncertainty_rejection(
    predictions: Tensor,
    targets: Tensor,
    uncertainty: Tensor,
    retained_fraction: float,
) -> tuple[Tensor, Tensor]:
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained fraction must be in (0, 1]")
    count = max(1, int(predictions.shape[0] * retained_fraction))
    indices = torch.argsort(uncertainty.flatten())[:count]
    return predictions[indices], targets[indices]

