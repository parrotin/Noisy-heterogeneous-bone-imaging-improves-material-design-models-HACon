from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from hacon.metrics.classification import macro_auc


@dataclass(frozen=True)
class ProbeResult:
    layer: str
    scanner_auc: float
    anatomy_auc: float
    scanner_accuracy: float
    anatomy_accuracy: float


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, classes: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.normalization(features))


def train_probe(
    probe: LinearProbe,
    features: Tensor,
    labels: Tensor,
    optimizer: Optimizer,
    epochs: int,
    batch_size: int = 256,
) -> LinearProbe:
    device = next(probe.parameters()).device
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = probe(batch_features)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return probe


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    features: Tensor,
    labels: Tensor,
) -> tuple[float, float]:
    device = next(probe.parameters()).device
    probe.eval()
    logits = probe(features.to(device))
    probabilities = torch.softmax(logits, dim=1)
    accuracy = (logits.argmax(dim=1) == labels.to(device)).float().mean()
    auc = macro_auc(probabilities, labels.to(device), probabilities.shape[1])
    return float(auc.cpu().item()), float(accuracy.cpu().item())


def layerwise_probe(
    layer_features: dict[str, Tensor],
    scanner_labels: Tensor,
    anatomy_labels: Tensor,
    scanner_classes: int,
    anatomy_classes: int,
    device: torch.device,
    epochs: int = 50,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for layer, features in layer_features.items():
        scanner_probe = LinearProbe(features.shape[1], scanner_classes).to(device)
        anatomy_probe = LinearProbe(features.shape[1], anatomy_classes).to(device)
        scanner_optimizer = torch.optim.AdamW(scanner_probe.parameters(), lr=1e-3)
        anatomy_optimizer = torch.optim.AdamW(anatomy_probe.parameters(), lr=1e-3)
        train_probe(scanner_probe, features, scanner_labels, scanner_optimizer, epochs)
        train_probe(anatomy_probe, features, anatomy_labels, anatomy_optimizer, epochs)
        scanner_auc, scanner_accuracy = evaluate_probe(scanner_probe, features, scanner_labels)
        anatomy_auc, anatomy_accuracy = evaluate_probe(anatomy_probe, features, anatomy_labels)
        results.append(
            ProbeResult(
                layer,
                scanner_auc,
                anatomy_auc,
                scanner_accuracy,
                anatomy_accuracy,
            )
        )
    return results


def covariance_matrix(features: Tensor) -> Tensor:
    centered = features - features.mean(dim=0)
    return centered.transpose(0, 1) @ centered / max(1, features.shape[0] - 1)


def effective_rank(features: Tensor) -> Tensor:
    singular_values = torch.linalg.svdvals(covariance_matrix(features))
    probabilities = singular_values / singular_values.sum().clamp_min(1e-7)
    entropy = -(probabilities * probabilities.clamp_min(1e-7).log()).sum()
    return entropy.exp()


def participation_ratio(features: Tensor) -> Tensor:
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix(features)).clamp_min(0.0)
    return eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-7)


def feature_uniformity(features: Tensor, temperature: float = 2.0) -> Tensor:
    normalized = torch.nn.functional.normalize(features, dim=1)
    distances = torch.pdist(normalized).square()
    return torch.log(torch.exp(-temperature * distances).mean())


def feature_alignment(first: Tensor, second: Tensor) -> Tensor:
    first_normalized = torch.nn.functional.normalize(first, dim=1)
    second_normalized = torch.nn.functional.normalize(second, dim=1)
    return (first_normalized - second_normalized).square().sum(dim=1).mean()


def centered_kernel_alignment(first: Tensor, second: Tensor) -> Tensor:
    first_gram = first @ first.transpose(0, 1)
    second_gram = second @ second.transpose(0, 1)
    size = first.shape[0]
    identity = torch.eye(size, device=first.device)
    centering = identity - torch.ones_like(identity) / size
    first_centered = centering @ first_gram @ centering
    second_centered = centering @ second_gram @ centering
    numerator = (first_centered * second_centered).sum()
    denominator = first_centered.square().sum().sqrt() * second_centered.square().sum().sqrt()
    return numerator / denominator.clamp_min(1e-7)


def class_centroids(features: Tensor, labels: Tensor) -> dict[int, Tensor]:
    return {
        int(label): features[labels == label].mean(dim=0)
        for label in torch.unique(labels).tolist()
    }


def within_class_scatter(features: Tensor, labels: Tensor) -> Tensor:
    centroids = class_centroids(features, labels)
    total = torch.zeros((), device=features.device)
    for label, centroid in centroids.items():
        selected = features[labels == label]
        total = total + (selected - centroid).square().sum()
    return total / features.shape[0]


def between_class_scatter(features: Tensor, labels: Tensor) -> Tensor:
    global_mean = features.mean(dim=0)
    centroids = class_centroids(features, labels)
    total = torch.zeros((), device=features.device)
    for label, centroid in centroids.items():
        count = (labels == label).sum()
        total = total + count * (centroid - global_mean).square().sum()
    return total / features.shape[0]


def fisher_discriminant_ratio(features: Tensor, labels: Tensor) -> Tensor:
    return between_class_scatter(features, labels) / within_class_scatter(features, labels).clamp_min(1e-7)

