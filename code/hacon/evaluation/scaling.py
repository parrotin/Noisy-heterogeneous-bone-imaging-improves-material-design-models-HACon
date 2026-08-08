from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScalingFit:
    intercept: float
    logarithmic_coefficient: float
    r_squared: float
    predicted_saturation_sites: float


@dataclass(frozen=True)
class DiversityPoint:
    sites: int
    volumes_per_site: int
    total_volumes: int
    score: float


def fit_logarithmic_scaling(sites: Tensor, scores: Tensor) -> ScalingFit:
    transformed = torch.log2(sites.float())
    design = torch.stack((torch.ones_like(transformed), transformed), dim=1)
    solution = torch.linalg.lstsq(design, scores.float()).solution
    prediction = design @ solution
    residual = (scores - prediction).square().sum()
    total = (scores - scores.mean()).square().sum()
    r_squared = 1.0 - residual / total.clamp_min(1e-7)
    coefficient = float(solution[1].item())
    saturation = float("inf") if coefficient <= 0.0 else 2.0 ** max(0.0, (1.0 - float(solution[0])) / coefficient)
    return ScalingFit(
        intercept=float(solution[0].item()),
        logarithmic_coefficient=coefficient,
        r_squared=float(r_squared.item()),
        predicted_saturation_sites=saturation,
    )


def predict_scaling(fit: ScalingFit, sites: Tensor) -> Tensor:
    return fit.intercept + fit.logarithmic_coefficient * torch.log2(sites.float())


def marginal_site_value(fit: ScalingFit, sites: int) -> float:
    current = fit.intercept + fit.logarithmic_coefficient * torch.log2(torch.tensor(float(sites)))
    next_value = fit.intercept + fit.logarithmic_coefficient * torch.log2(torch.tensor(float(sites + 1)))
    return float((next_value - current).item())


def diversity_advantage(diverse: DiversityPoint, homogeneous: DiversityPoint) -> float:
    if diverse.total_volumes != homogeneous.total_volumes:
        raise ValueError("diversity comparison requires equal volume counts")
    return diverse.score - homogeneous.score


def site_diversity_bound(
    sites: Tensor,
    scanner_subspace_dimension: float,
    representation_dimension: float,
    capacity: float,
    pair_budget: Tensor,
) -> Tensor:
    first = torch.log(sites.float()) / (sites.float() - 1.0)
    dimension_ratio = scanner_subspace_dimension / representation_dimension
    complexity = capacity / pair_budget.float().sqrt()
    return first * dimension_ratio + complexity


def positive_pair_types(sites: int) -> int:
    return sites * (sites - 1)


def total_pair_budget(regions: int, sites: int, pairs_per_type: int) -> int:
    return regions * positive_pair_types(sites) * pairs_per_type


def fixed_volume_design(
    total_volumes: int,
    site_counts: list[int],
    measured_scores: list[float],
) -> list[DiversityPoint]:
    if len(site_counts) != len(measured_scores):
        raise ValueError("site counts and scores must align")
    return [
        DiversityPoint(
            sites=sites,
            volumes_per_site=total_volumes // sites,
            total_volumes=total_volumes,
            score=score,
        )
        for sites, score in zip(site_counts, measured_scores)
    ]


def label_efficiency_ratio(
    reference_fraction: float,
    method_fraction: float,
) -> float:
    if method_fraction <= 0.0:
        raise ValueError("method label fraction must be positive")
    return reference_fraction / method_fraction


def normalized_compute_efficiency(score: float, gpu_hours: float) -> float:
    if gpu_hours <= 0.0:
        raise ValueError("GPU hours must be positive")
    return score / gpu_hours


def relative_compute_reduction(reference_hours: float, method_hours: float) -> float:
    return (reference_hours - method_hours) / reference_hours


def throughput(volumes: int, epochs: int, gpu_hours: float, gpus: int) -> float:
    total_examples = volumes * epochs
    device_hours = gpu_hours
    if device_hours <= 0.0 or gpus <= 0:
        raise ValueError("compute values must be positive")
    return total_examples / (device_hours * 3600.0)

