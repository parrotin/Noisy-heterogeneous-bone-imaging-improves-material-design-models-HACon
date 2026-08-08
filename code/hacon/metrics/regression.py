from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RegressionMetrics:
    r2: Tensor
    mae: Tensor
    rmse: Tensor
    relative_error: Tensor
    pearson: Tensor
    spearman: Tensor


def mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(prediction - target), dim=0)


def mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target).square(), dim=0)


def root_mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    return mean_squared_error(prediction, target).sqrt()


def relative_error(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    return torch.mean(torch.abs(prediction - target) / target.abs().clamp_min(epsilon), dim=0)


def r2_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    residual = (target - prediction).square().sum(dim=0)
    centered = (target - target.mean(dim=0)).square().sum(dim=0)
    return 1.0 - residual / centered.clamp_min(epsilon)


def pearson_correlation(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    first = prediction - prediction.mean(dim=0)
    second = target - target.mean(dim=0)
    numerator = (first * second).sum(dim=0)
    denominator = first.square().sum(dim=0).sqrt() * second.square().sum(dim=0).sqrt()
    return numerator / denominator.clamp_min(epsilon)


def rank_data(values: Tensor) -> Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.argsort(order, dim=0).float()
    return ranks


def spearman_correlation(prediction: Tensor, target: Tensor) -> Tensor:
    return pearson_correlation(rank_data(prediction), rank_data(target))


def regression_metrics(prediction: Tensor, target: Tensor) -> RegressionMetrics:
    return RegressionMetrics(
        r2=r2_score(prediction, target),
        mae=mean_absolute_error(prediction, target),
        rmse=root_mean_squared_error(prediction, target),
        relative_error=relative_error(prediction, target),
        pearson=pearson_correlation(prediction, target),
        spearman=spearman_correlation(prediction, target),
    )


def concordance_correlation(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    prediction_mean = prediction.mean(dim=0)
    target_mean = target.mean(dim=0)
    prediction_var = prediction.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    covariance = ((prediction - prediction_mean) * (target - target_mean)).mean(dim=0)
    denominator = prediction_var + target_var + (prediction_mean - target_mean).square()
    return 2.0 * covariance / denominator.clamp_min(epsilon)


def maximum_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.abs(prediction - target).max(dim=0).values


def median_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.abs(prediction - target).median(dim=0).values


def mean_absolute_percentage_error(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    return (torch.abs(prediction - target) / target.abs().clamp_min(epsilon)).mean(dim=0) * 100.0


def symmetric_percentage_error(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    numerator = 2.0 * torch.abs(prediction - target)
    denominator = prediction.abs() + target.abs()
    return (numerator / denominator.clamp_min(epsilon)).mean(dim=0) * 100.0


def explained_variance(prediction: Tensor, target: Tensor, epsilon: float = 1e-7) -> Tensor:
    residual = target - prediction
    return 1.0 - residual.var(dim=0, unbiased=False) / target.var(dim=0, unbiased=False).clamp_min(epsilon)


class TargetNormalizer:
    def __init__(self) -> None:
        self.mean: Tensor | None = None
        self.standard_deviation: Tensor | None = None

    def fit(self, target: Tensor) -> TargetNormalizer:
        self.mean = target.mean(dim=0)
        self.standard_deviation = target.std(dim=0).clamp_min(1e-7)
        return self

    def transform(self, target: Tensor) -> Tensor:
        if self.mean is None or self.standard_deviation is None:
            raise RuntimeError("normalizer has not been fitted")
        return (target - self.mean) / self.standard_deviation

    def inverse_transform(self, target: Tensor) -> Tensor:
        if self.mean is None or self.standard_deviation is None:
            raise RuntimeError("normalizer has not been fitted")
        return target * self.standard_deviation + self.mean

