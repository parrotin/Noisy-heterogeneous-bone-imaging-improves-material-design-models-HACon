from __future__ import annotations

import torch
from torch import Tensor, nn

from hacon.models.blocks import ProjectionHead, ResidualStage


class ResNet50Volume(nn.Module):
    def __init__(self, input_channels: int = 1, representation_dim: int = 512) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.stage1 = ResidualStage(64, 64, 3, 1)
        self.stage2 = ResidualStage(256, 128, 4, 2)
        self.stage3 = ResidualStage(512, 256, 6, 2)
        self.stage4 = ResidualStage(1024, 512, 3, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embedding = nn.Linear(2048, representation_dim)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        first = self.stage1(x)
        second = self.stage2(first)
        third = self.stage3(second)
        fourth = self.stage4(third)
        return first, second, third, fourth

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        pooled = self.pool(features[-1]).flatten(1)
        return self.embedding(pooled)


class HAConEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        representation_dim: int = 512,
        projection_hidden_dim: int = 2048,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Volume(input_channels, representation_dim)
        self.projector = ProjectionHead(
            representation_dim,
            projection_hidden_dim,
            projection_dim,
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def project(self, representation: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.projector(representation), dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        representation = self.encode(x)
        projection = self.project(representation)
        return representation, projection


class FrozenFeatureExtractor(nn.Module):
    def __init__(self, encoder: HAConEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> FrozenFeatureExtractor:
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.encoder.encode(x)

