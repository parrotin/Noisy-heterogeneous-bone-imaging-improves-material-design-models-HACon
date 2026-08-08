from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hacon.metrics.regression import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class DownstreamEpoch:
    epoch: int
    training_loss: float
    validation: RegressionMetrics | None


class MaterialTrainer:
    def __init__(
        self,
        extractor: nn.Module,
        head: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        loss: nn.Module | None = None,
    ) -> None:
        self.extractor = extractor.to(device)
        self.head = head.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss = nn.MSELoss() if loss is None else loss

    def train_epoch(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> float:
        self.extractor.eval()
        self.head.train()
        losses: list[float] = []
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                features = self.extractor(images)
            predictions = self.head(features)
            loss = self.loss(predictions, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return sum(losses) / len(losses)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> RegressionMetrics:
        self.extractor.eval()
        self.head.eval()
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        for images, target in loader:
            images = images.to(self.device, non_blocking=True)
            predictions.append(self.head(self.extractor(images)).cpu())
            targets.append(target.cpu())
        return regression_metrics(torch.cat(predictions), torch.cat(targets))

    def fit(
        self,
        training_loader: DataLoader[tuple[Tensor, Tensor]],
        validation_loader: DataLoader[tuple[Tensor, Tensor]] | None,
        epochs: int,
    ) -> list[DownstreamEpoch]:
        history: list[DownstreamEpoch] = []
        for epoch in range(epochs):
            training_loss = self.train_epoch(training_loader)
            validation = None
            if validation_loader is not None:
                validation = self.evaluate(validation_loader)
            history.append(DownstreamEpoch(epoch, training_loss, validation))
        return history


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def combined_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).flatten(1).sum(dim=1)
        denominator = probabilities.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        binary_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return dice_loss + binary_cross_entropy

    def train_epoch(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> float:
        self.model.train()
        losses: list[float] = []
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.combined_loss(logits, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return sum(losses) / len(losses)

