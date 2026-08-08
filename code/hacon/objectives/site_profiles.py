from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SiteProfile:
    site_id: int
    hu_offset: float
    noise_power_spectrum: Tensor
    resolution: float


@dataclass(frozen=True)
class NormalizationBounds:
    hu_min: float
    hu_max: float
    nps_min: float
    nps_max: float
    resolution_min: float
    resolution_max: float


def _range(values: Tensor) -> tuple[float, float]:
    return float(values.min().item()), float(values.max().item())


def _normalize(value: Tensor, minimum: float, maximum: float) -> Tensor:
    width = maximum - minimum
    if width <= 0.0:
        return torch.zeros_like(value)
    return (value - minimum) / width


def component_matrices(profiles: list[SiteProfile]) -> tuple[Tensor, Tensor, Tensor]:
    if len(profiles) < 2:
        raise ValueError("at least two site profiles are required")
    hu = torch.tensor([profile.hu_offset for profile in profiles], dtype=torch.float64)
    resolution = torch.tensor([profile.resolution for profile in profiles], dtype=torch.float64)
    nps = torch.stack([profile.noise_power_spectrum.double() for profile in profiles])
    hu_distance = torch.abs(hu[:, None] - hu[None, :])
    resolution_distance = torch.abs(resolution[:, None] / resolution[None, :] - 1.0)
    nps_distance = torch.cdist(nps, nps, p=2)
    return hu_distance, nps_distance, resolution_distance


def scanner_dissimilarity(profiles: list[SiteProfile]) -> Tensor:
    hu, nps, resolution = component_matrices(profiles)
    hu_norm = _normalize(hu, *_range(hu))
    nps_norm = _normalize(nps, *_range(nps))
    resolution_norm = _normalize(resolution, *_range(resolution))
    distance = hu_norm + nps_norm + resolution_norm
    distance.fill_diagonal_(0.0)
    return distance.float()


def site_temperature(
    distance: Tensor,
    base_temperature: float = 0.07,
    modulation_strength: float = 1.5,
) -> Tensor:
    if base_temperature <= 0.0:
        raise ValueError("base temperature must be positive")
    if modulation_strength < 0.0:
        raise ValueError("modulation strength cannot be negative")
    return base_temperature * torch.exp(-modulation_strength * distance)


def estimate_air_offset(volume: Tensor, air_mask: Tensor) -> float:
    selected = volume[air_mask.bool()]
    if selected.numel() == 0:
        raise ValueError("air mask is empty")
    return float(selected.mean().item())


def radial_noise_power_spectrum(
    volume: Tensor,
    homogeneous_mask: Tensor,
    bins: int = 64,
) -> Tensor:
    selected = volume * homogeneous_mask
    centered = selected - selected[homogeneous_mask.bool()].mean()
    spectrum = torch.fft.fftn(centered.double())
    power = torch.fft.fftshift(spectrum.abs().square())
    axes = [
        torch.arange(size, device=volume.device, dtype=torch.float64) - size // 2
        for size in volume.shape[-3:]
    ]
    grid = torch.meshgrid(*axes, indexing="ij")
    radius = torch.sqrt(sum(axis.square() for axis in grid))
    scaled = radius / radius.max().clamp_min(1.0) * (bins - 1)
    indices = scaled.long().clamp(0, bins - 1)
    totals = torch.zeros(bins, dtype=torch.float64, device=volume.device)
    counts = torch.zeros(bins, dtype=torch.float64, device=volume.device)
    totals.scatter_add_(0, indices.flatten(), power.flatten())
    counts.scatter_add_(0, indices.flatten(), torch.ones_like(power).flatten())
    return (totals / counts.clamp_min(1.0)).float()

