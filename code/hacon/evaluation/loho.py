from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor


MetricT = TypeVar("MetricT")


@dataclass(frozen=True)
class SiteFold(Generic[MetricT]):
    held_out_site: str
    in_distribution: MetricT
    out_distribution: MetricT
    gap: MetricT
    test_count: int


@dataclass(frozen=True)
class AggregateFold:
    mean_id: float
    mean_ood: float
    mean_gap: float
    standard_deviation_ood: float
    sites: int
    test_count: int


def make_site_fold(
    held_out_site: str,
    in_distribution: float,
    out_distribution: float,
    test_count: int,
) -> SiteFold[float]:
    return SiteFold(
        held_out_site,
        in_distribution,
        out_distribution,
        in_distribution - out_distribution,
        test_count,
    )


def aggregate_folds(folds: list[SiteFold[float]]) -> AggregateFold:
    if not folds:
        raise ValueError("at least one fold is required")
    id_values = torch.tensor([fold.in_distribution for fold in folds])
    ood_values = torch.tensor([fold.out_distribution for fold in folds])
    gaps = torch.tensor([fold.gap for fold in folds])
    return AggregateFold(
        mean_id=float(id_values.mean().item()),
        mean_ood=float(ood_values.mean().item()),
        mean_gap=float(gaps.mean().item()),
        standard_deviation_ood=float(ood_values.std(unbiased=True).item()),
        sites=len(folds),
        test_count=sum(fold.test_count for fold in folds),
    )


def weighted_aggregate_folds(folds: list[SiteFold[float]]) -> AggregateFold:
    if not folds:
        raise ValueError("at least one fold is required")
    weights = torch.tensor([fold.test_count for fold in folds], dtype=torch.float64)
    weights = weights / weights.sum()
    id_values = torch.tensor([fold.in_distribution for fold in folds], dtype=torch.float64)
    ood_values = torch.tensor([fold.out_distribution for fold in folds], dtype=torch.float64)
    gaps = torch.tensor([fold.gap for fold in folds], dtype=torch.float64)
    mean_ood = (ood_values * weights).sum()
    variance = ((ood_values - mean_ood).square() * weights).sum()
    return AggregateFold(
        mean_id=float((id_values * weights).sum().item()),
        mean_ood=float(mean_ood.item()),
        mean_gap=float((gaps * weights).sum().item()),
        standard_deviation_ood=float(variance.sqrt().item()),
        sites=len(folds),
        test_count=sum(fold.test_count for fold in folds),
    )


def vendor_partition(
    training_vendors: set[str],
    sample_vendors: list[str],
) -> tuple[Tensor, Tensor]:
    id_mask = torch.tensor([vendor in training_vendors for vendor in sample_vendors])
    return id_mask, ~id_mask


def domain_scores(
    scores: Tensor,
    id_mask: Tensor,
    ood_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    id_score = scores[id_mask].mean()
    ood_score = scores[ood_mask].mean()
    return id_score, ood_score, id_score - ood_score


def seed_matrix(results: dict[str, list[float]], ordered_sites: list[str]) -> Tensor:
    rows = [results[site] for site in ordered_sites]
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError("all sites require equal seed counts")
    return torch.tensor(rows, dtype=torch.float64)


def site_consistency(seed_results: Tensor) -> Tensor:
    site_means = seed_results.mean(dim=1)
    return 1.0 - site_means.std(unbiased=True) / site_means.mean().clamp_min(1e-7)


def worst_site_score(seed_results: Tensor) -> Tensor:
    return seed_results.mean(dim=1).min()


def best_site_score(seed_results: Tensor) -> Tensor:
    return seed_results.mean(dim=1).max()


def site_range(seed_results: Tensor) -> Tensor:
    means = seed_results.mean(dim=1)
    return means.max() - means.min()

