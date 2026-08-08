from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm3d,
    ) -> None:
        super().__init__()
        width = channels
        self.conv1 = conv1x1(in_channels, width)
        self.norm1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.norm2 = norm_layer(width)
        self.conv3 = conv1x1(width, channels * self.expansion)
        self.norm3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        blocks: int,
        stride: int,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm3d,
    ) -> None:
        super().__init__()
        out_channels = channels * Bottleneck3D.expansion
        downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels),
            )
        layers: list[nn.Module] = [
            Bottleneck3D(in_channels, channels, stride, downsample, norm_layer)
        ]
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(out_channels, channels, 1, None, norm_layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SegmentationDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], classes: int) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up4 = nn.ConvTranspose3d(c4, c3, 2, 2)
        self.refine4 = self._refine(c3 + c3, c3)
        self.up3 = nn.ConvTranspose3d(c3, c2, 2, 2)
        self.refine3 = self._refine(c2 + c2, c2)
        self.up2 = nn.ConvTranspose3d(c2, c1, 2, 2)
        self.refine2 = self._refine(c1 + c1, c1)
        self.up1 = nn.ConvTranspose3d(c1, 64, 2, 2)
        self.refine1 = self._refine(64, 64)
        self.output = nn.Conv3d(64, classes, 1)

    def _refine(self, input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def _resize(self, x: Tensor, target: Tensor) -> Tensor:
        shape = target.shape[-3:]
        return torch.nn.functional.interpolate(x, size=shape, mode="trilinear", align_corners=False)

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f1, f2, f3, f4 = features
        x = self._resize(self.up4(f4), f3)
        x = self.refine4(torch.cat((x, f3), dim=1))
        x = self._resize(self.up3(x), f2)
        x = self.refine3(torch.cat((x, f2), dim=1))
        x = self._resize(self.up2(x), f1)
        x = self.refine2(torch.cat((x, f1), dim=1))
        x = self.refine1(self.up1(x))
        return self.output(x)

