from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import Tensor


@dataclass(frozen=True)
class Summary:
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_lower: float
    confidence_upper: float
    count: int


@dataclass(frozen=True)
class PairedComparison:
    mean_difference: float
    statistic: float
    p_value: float
    effect_size: float
    confidence_lower: float
    confidence_upper: float
    count: int


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float
    samples: int


def summarize(values: Tensor, confidence: float = 0.95) -> Summary:
    array = values.detach().double().cpu().numpy()
    count = int(array.size)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if count > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(count)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, max(1, count - 1)))
    margin = critical * standard_error
    return Summary(
        mean,
        standard_deviation,
        standard_error,
        mean - margin,
        mean + margin,
        count,
    )


def paired_t_test(first: Tensor, second: Tensor, confidence: float = 0.95) -> PairedComparison:
    if first.shape != second.shape:
        raise ValueError("paired samples must have equal shape")
    first_array = first.detach().double().cpu().numpy()
    second_array = second.detach().double().cpu().numpy()
    difference = first_array - second_array
    result = stats.ttest_rel(first_array, second_array)
    count = difference.size
    mean_difference = float(np.mean(difference))
    deviation = float(np.std(difference, ddof=1))
    effect_size = mean_difference / deviation if deviation > 0.0 else float("inf")
    error = deviation / math.sqrt(count)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, count - 1))
    margin = critical * error
    return PairedComparison(
        mean_difference=mean_difference,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=effect_size,
        confidence_lower=mean_difference - margin,
        confidence_upper=mean_difference + margin,
        count=count,
    )


def cohen_d_independent(first: Tensor, second: Tensor) -> float:
    first_array = first.detach().double().cpu().numpy()
    second_array = second.detach().double().cpu().numpy()
    first_variance = np.var(first_array, ddof=1)
    second_variance = np.var(second_array, ddof=1)
    degrees = first_array.size + second_array.size - 2
    pooled = math.sqrt(
        ((first_array.size - 1) * first_variance + (second_array.size - 1) * second_variance)
        / degrees
    )
    return float((np.mean(first_array) - np.mean(second_array)) / pooled)


def bootstrap_mean(
    values: Tensor,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 17,
) -> BootstrapInterval:
    generator = torch.Generator(device=values.device).manual_seed(seed)
    indices = torch.randint(values.numel(), (samples, values.numel()), generator=generator, device=values.device)
    estimates = values.flatten()[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(values.mean().item()),
        lower=float(torch.quantile(estimates, alpha).item()),
        upper=float(torch.quantile(estimates, 1.0 - alpha).item()),
        standard_error=float(estimates.std(unbiased=True).item()),
        samples=samples,
    )


def bootstrap_metric(
    prediction: Tensor,
    target: Tensor,
    metric: callable,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 17,
) -> BootstrapInterval:
    generator = torch.Generator(device=prediction.device).manual_seed(seed)
    count = prediction.shape[0]
    estimates: list[Tensor] = []
    for _ in range(samples):
        indices = torch.randint(count, (count,), generator=generator, device=prediction.device)
        estimates.append(metric(prediction[indices], target[indices]))
    distribution = torch.stack(estimates).float()
    estimate = metric(prediction, target)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(estimate.item()),
        lower=float(torch.quantile(distribution, alpha).item()),
        upper=float(torch.quantile(distribution, 1.0 - alpha).item()),
        standard_error=float(distribution.std(unbiased=True).item()),
        samples=samples,
    )


def permutation_test(
    first: Tensor,
    second: Tensor,
    permutations: int = 10000,
    seed: int = 17,
) -> float:
    if first.shape != second.shape:
        raise ValueError("samples must have equal shape")
    difference = first - second
    observed = difference.mean().abs()
    generator = torch.Generator(device=difference.device).manual_seed(seed)
    exceedances = 0
    for _ in range(permutations):
        signs = torch.randint(0, 2, difference.shape, generator=generator, device=difference.device)
        signs = signs.float() * 2.0 - 1.0
        statistic = (difference * signs).mean().abs()
        exceedances += int(statistic >= observed)
    return (exceedances + 1) / (permutations + 1)


def interaction_ratio(
    full: float,
    without_first: float,
    without_second: float,
    without_both: float,
) -> float:
    first_drop = full - without_first
    second_drop = full - without_second
    combined_drop = full - without_both
    denominator = first_drop + second_drop
    if denominator == 0.0:
        return float("nan")
    return combined_drop / denominator


def cross_site_gap(in_distribution: Tensor, out_distribution: Tensor) -> Tensor:
    if in_distribution.shape != out_distribution.shape:
        raise ValueError("ID and OOD observations must align")
    return in_distribution - out_distribution


def gap_reduction(reference_gap: Tensor, method_gap: Tensor) -> Tensor:
    return (reference_gap - method_gap) / reference_gap.clamp_min(1e-7)


def benjamini_hochberg(p_values: Tensor, alpha: float = 0.05) -> Tensor:
    order = torch.argsort(p_values)
    sorted_values = p_values[order]
    count = p_values.numel()
    thresholds = torch.arange(1, count + 1, device=p_values.device) / count * alpha
    valid = sorted_values <= thresholds
    rejected = torch.zeros_like(valid)
    if valid.any():
        maximum = torch.nonzero(valid, as_tuple=False).max()
        rejected[: maximum + 1] = True
    result = torch.zeros_like(rejected)
    result[order] = rejected
    return result


def holm_bonferroni(p_values: Tensor, alpha: float = 0.05) -> Tensor:
    order = torch.argsort(p_values)
    sorted_values = p_values[order]
    count = p_values.numel()
    rejected = torch.zeros(count, dtype=torch.bool, device=p_values.device)
    for index, value in enumerate(sorted_values):
        if value <= alpha / (count - index):
            rejected[index] = True
        else:
            break
    result = torch.zeros_like(rejected)
    result[order] = rejected
    return result


def fisher_z_interval(correlation: float, count: int, confidence: float = 0.95) -> tuple[float, float]:
    if count <= 3:
        raise ValueError("at least four observations are required")
    transformed = np.arctanh(correlation)
    error = 1.0 / math.sqrt(count - 3)
    critical = stats.norm.ppf((1.0 + confidence) / 2.0)
    return float(np.tanh(transformed - critical * error)), float(np.tanh(transformed + critical * error))

