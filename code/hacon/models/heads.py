from __future__ import annotations

import torch
from torch import Tensor, nn


class MaterialPropertyHead(nn.Module):
    def __init__(self, input_dim: int = 512, outputs: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ScannerProbe(nn.Module):
    def __init__(self, input_dim: int, sites: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, sites)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.normalization(x))


class AnatomyProbe(nn.Module):
    def __init__(self, input_dim: int, regions: int = 59) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, regions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class DeepEnsemble(nn.Module):
    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("at least one ensemble member is required")
        self.members = nn.ModuleList(members)

    def forward_members(self, x: Tensor) -> Tensor:
        return torch.stack([member(x) for member in self.members], dim=0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.forward_members(x)
        return predictions.mean(dim=0), predictions.var(dim=0, unbiased=False)


class MultiViewAggregator(nn.Module):
    def __init__(self, encoder: nn.Module, reducer: str = "mean") -> None:
        super().__init__()
        if reducer not in {"mean", "median", "maximum"}:
            raise ValueError("unsupported reducer")
        self.encoder = encoder
        self.reducer = reducer

    def forward(self, views: Tensor) -> Tensor:
        batch, count = views.shape[:2]
        flat = views.reshape(batch * count, *views.shape[2:])
        encoded = self.encoder(flat).reshape(batch, count, -1)
        if self.reducer == "mean":
            return encoded.mean(dim=1)
        if self.reducer == "median":
            return encoded.median(dim=1).values
        return encoded.max(dim=1).values

