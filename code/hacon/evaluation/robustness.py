from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional


class Perturbation(Protocol):
    def __call__(self, volume: Tensor) -> Tensor:
        ...


@dataclass(frozen=True)
class RobustnessPoint:
    name: str
    severity: float
    score: float
    degradation: float


class AdditiveGaussian:
    def __init__(self, standard_deviation: float) -> None:
        self.standard_deviation = standard_deviation

    def __call__(self, volume: Tensor) -> Tensor:
        return (volume + torch.randn_like(volume) * self.standard_deviation).clamp(0.0, 1.0)


class HounsfieldOffset:
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def __call__(self, volume: Tensor) -> Tensor:
        return (volume + self.fraction).clamp(0.0, 1.0)


class HounsfieldScale:
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def __call__(self, volume: Tensor) -> Tensor:
        return (volume * (1.0 + self.fraction)).clamp(0.0, 1.0)


class ResolutionDegradation:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, volume: Tensor) -> Tensor:
        shape = volume.shape[-3:]
        reduced = tuple(max(1, int(size / self.factor)) for size in shape)
        down = functional.interpolate(volume, size=reduced, mode="trilinear", align_corners=False)
        return functional.interpolate(down, size=shape, mode="trilinear", align_corners=False)


class KernelBlur:
    def __init__(self, radius: int) -> None:
        self.radius = radius

    def __call__(self, volume: Tensor) -> Tensor:
        size = self.radius * 2 + 1
        return functional.avg_pool3d(volume, size, stride=1, padding=self.radius)


class SliceThickness:
    def __init__(self, factor: int, axis: int = -1) -> None:
        self.factor = factor
        self.axis = axis

    def __call__(self, volume: Tensor) -> Tensor:
        indices = torch.arange(0, volume.shape[self.axis], self.factor, device=volume.device)
        selected = torch.index_select(volume, self.axis, indices)
        return functional.interpolate(selected, size=volume.shape[-3:], mode="trilinear", align_corners=False)


class MotionGhost:
    def __init__(self, displacement: int, strength: float = 0.2) -> None:
        self.displacement = displacement
        self.strength = strength

    def __call__(self, volume: Tensor) -> Tensor:
        ghost = torch.roll(volume, self.displacement, dims=-1)
        return (volume * (1.0 - self.strength) + ghost * self.strength).clamp(0.0, 1.0)


class RingArtifact:
    def __init__(self, strength: float = 0.05, frequency: float = 20.0) -> None:
        self.strength = strength
        self.frequency = frequency

    def __call__(self, volume: Tensor) -> Tensor:
        height, width = volume.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=volume.device)
        x = torch.linspace(-1.0, 1.0, width, device=volume.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        rings = torch.sin(radius * self.frequency) * self.strength
        return (volume + rings).clamp(0.0, 1.0)


class MetalStreak:
    def __init__(self, strength: float = 0.15, lines: int = 8) -> None:
        self.strength = strength
        self.lines = lines

    def __call__(self, volume: Tensor) -> Tensor:
        result = volume.clone()
        width = volume.shape[-1]
        locations = torch.linspace(0, width - 1, self.lines).long()
        for index, location in enumerate(locations.tolist()):
            offset = (index % 3) - 1
            result[..., max(0, location + offset):min(width, location + offset + 2)] += self.strength
        return result.clamp(0.0, 1.0)


class PoissonNoise:
    def __init__(self, scale: float = 100.0) -> None:
        self.scale = scale

    def __call__(self, volume: Tensor) -> Tensor:
        return torch.poisson(volume * self.scale) / self.scale


class SaltPepper:
    def __init__(self, probability: float = 0.01) -> None:
        self.probability = probability

    def __call__(self, volume: Tensor) -> Tensor:
        random = torch.rand_like(volume)
        result = volume.clone()
        result[random < self.probability / 2.0] = 0.0
        result[random > 1.0 - self.probability / 2.0] = 1.0
        return result


class BiasGradient:
    def __init__(self, strength: float = 0.1, axis: int = -1) -> None:
        self.strength = strength
        self.axis = axis

    def __call__(self, volume: Tensor) -> Tensor:
        shape = [1] * volume.ndim
        shape[self.axis] = volume.shape[self.axis]
        gradient = torch.linspace(1.0 - self.strength, 1.0 + self.strength, volume.shape[self.axis], device=volume.device).reshape(shape)
        return (volume * gradient).clamp(0.0, 1.0)


class RobustnessEvaluator:
    def __init__(self, model: nn.Module, metric: callable, device: torch.device) -> None:
        self.model = model.to(device)
        self.metric = metric
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        images: Tensor,
        targets: Tensor,
        perturbations: list[tuple[str, float, Perturbation]],
    ) -> list[RobustnessPoint]:
        self.model.eval()
        images = images.to(self.device)
        targets = targets.to(self.device)
        baseline = float(self.metric(self.model(images), targets).mean().item())
        points: list[RobustnessPoint] = []
        for name, severity, perturbation in perturbations:
            prediction = self.model(perturbation(images))
            score = float(self.metric(prediction, targets).mean().item())
            points.append(RobustnessPoint(name, severity, score, baseline - score))
        return points


def standard_perturbations() -> list[tuple[str, float, Perturbation]]:
    result: list[tuple[str, float, Perturbation]] = []
    for severity in (0.01, 0.03, 0.05, 0.10):
        result.append(("gaussian_noise", severity, AdditiveGaussian(severity)))
    for severity in (-0.10, -0.05, 0.05, 0.10):
        result.append(("hu_offset", severity, HounsfieldOffset(severity)))
    for severity in (1.25, 1.5, 2.0):
        result.append(("resolution", severity, ResolutionDegradation(severity)))
    for severity in (1.0, 2.0, 3.0):
        result.append(("kernel_blur", severity, KernelBlur(int(severity))))
    return result

