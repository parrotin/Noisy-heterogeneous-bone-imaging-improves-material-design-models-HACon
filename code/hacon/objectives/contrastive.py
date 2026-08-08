from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hacon.objectives.pair_mining import PairSet
from hacon.objectives.site_profiles import site_temperature


@dataclass(frozen=True)
class ContrastiveOutput:
    loss: Tensor
    positive_similarity: Tensor
    negative_similarity: Tensor
    temperatures: Tensor
    pair_count: int


def cosine_matrix(projections: Tensor) -> Tensor:
    normalized = torch.nn.functional.normalize(projections, dim=-1)
    return normalized @ normalized.transpose(0, 1)


def pairwise_temperatures(
    pairs: PairSet,
    sites: Tensor,
    distance_matrix: Tensor,
    base_temperature: float,
    modulation_strength: float,
) -> Tensor:
    first_sites = sites[pairs.anchors]
    second_sites = sites[pairs.positives]
    distances = distance_matrix[first_sites, second_sites]
    return site_temperature(distances, base_temperature, modulation_strength)


class HeterogeneityAwareInfoNCE(nn.Module):
    def __init__(
        self,
        base_temperature: float = 0.07,
        modulation_strength: float = 1.5,
    ) -> None:
        super().__init__()
        self.base_temperature = base_temperature
        self.modulation_strength = modulation_strength

    def forward(
        self,
        projections: Tensor,
        sites: Tensor,
        pairs: PairSet,
        distance_matrix: Tensor,
    ) -> ContrastiveOutput:
        if pairs.count == 0:
            raise ValueError("the batch contains no valid positive pairs")
        similarities = cosine_matrix(projections)
        temperatures = pairwise_temperatures(
            pairs,
            sites,
            distance_matrix,
            self.base_temperature,
            self.modulation_strength,
        )
        positive = similarities[pairs.anchors, pairs.positives]
        logits = similarities[pairs.anchors] / temperatures[:, None]
        valid_negatives = pairs.negatives
        negative_logits = logits.masked_fill(~valid_negatives, float("-inf"))
        denominator = torch.logsumexp(
            torch.cat((positive[:, None] / temperatures[:, None], negative_logits), dim=1),
            dim=1,
        )
        loss = -(positive / temperatures - denominator).mean()
        negative_values = similarities[pairs.anchors[:, None], torch.arange(similarities.shape[0], device=similarities.device)]
        masked_negative = negative_values[valid_negatives]
        return ContrastiveOutput(
            loss=loss,
            positive_similarity=positive.mean(),
            negative_similarity=masked_negative.mean(),
            temperatures=temperatures,
            pair_count=pairs.count,
        )


class UniformInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: Tensor, pairs: PairSet) -> Tensor:
        if pairs.count == 0:
            raise ValueError("the batch contains no valid positive pairs")
        similarities = cosine_matrix(projections)
        positive = similarities[pairs.anchors, pairs.positives] / self.temperature
        negative = similarities[pairs.anchors] / self.temperature
        negative = negative.masked_fill(~pairs.negatives, float("-inf"))
        denominator = torch.logsumexp(torch.cat((positive[:, None], negative), dim=1), dim=1)
        return -(positive - denominator).mean()

