from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairSet:
    anchors: Tensor
    positives: Tensor
    negatives: Tensor

    @property
    def count(self) -> int:
        return int(self.anchors.numel())


def cross_site_positive_pairs(
    regions: Tensor,
    sites: Tensor,
    generator: torch.Generator | None = None,
) -> PairSet:
    if regions.ndim != 1 or sites.ndim != 1:
        raise ValueError("regions and sites must be one-dimensional")
    if regions.shape != sites.shape:
        raise ValueError("regions and sites must have equal shape")
    anchor_indices: list[int] = []
    positive_indices: list[int] = []
    for region in torch.unique(regions, sorted=True).tolist():
        group = torch.nonzero(regions == region, as_tuple=False).flatten()
        group_sites = torch.unique(sites[group], sorted=True)
        if group_sites.numel() < 2:
            continue
        for first_site in group_sites.tolist():
            for second_site in group_sites.tolist():
                if first_site == second_site:
                    continue
                first_group = group[sites[group] == first_site]
                second_group = group[sites[group] == second_site]
                first_pick = torch.randint(first_group.numel(), (1,), generator=generator)
                second_pick = torch.randint(second_group.numel(), (1,), generator=generator)
                anchor_indices.append(int(first_group[first_pick].item()))
                positive_indices.append(int(second_group[second_pick].item()))
    device = regions.device
    anchors = torch.tensor(anchor_indices, device=device, dtype=torch.long)
    positives = torch.tensor(positive_indices, device=device, dtype=torch.long)
    if anchors.numel() == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return PairSet(empty, empty, torch.empty((0, 0), device=device, dtype=torch.bool))
    negative_mask = regions[anchors, None] != regions[None, :]
    return PairSet(anchors, positives, negative_mask)


def within_site_positive_pairs(
    regions: Tensor,
    sites: Tensor,
    generator: torch.Generator | None = None,
) -> PairSet:
    anchors: list[int] = []
    positives: list[int] = []
    for region in torch.unique(regions, sorted=True).tolist():
        for site in torch.unique(sites, sorted=True).tolist():
            group = torch.nonzero((regions == region) & (sites == site), as_tuple=False).flatten()
            if group.numel() < 2:
                continue
            order = torch.randperm(group.numel(), generator=generator, device=group.device)
            usable = group[order]
            anchors.extend(usable.tolist())
            positives.extend(torch.roll(usable, shifts=1).tolist())
    device = regions.device
    anchor_tensor = torch.tensor(anchors, device=device, dtype=torch.long)
    positive_tensor = torch.tensor(positives, device=device, dtype=torch.long)
    if anchor_tensor.numel() == 0:
        return PairSet(
            anchor_tensor,
            positive_tensor,
            torch.empty((0, 0), device=device, dtype=torch.bool),
        )
    negative_mask = regions[anchor_tensor, None] != regions[None, :]
    return PairSet(anchor_tensor, positive_tensor, negative_mask)


def all_cross_site_matches(regions: Tensor, sites: Tensor) -> PairSet:
    region_match = regions[:, None] == regions[None, :]
    site_mismatch = sites[:, None] != sites[None, :]
    identity = torch.eye(regions.numel(), device=regions.device, dtype=torch.bool)
    positive_mask = region_match & site_mismatch & ~identity
    anchors, positives = torch.nonzero(positive_mask, as_tuple=True)
    negatives = regions[anchors, None] != regions[None, :]
    return PairSet(anchors, positives, negatives)

