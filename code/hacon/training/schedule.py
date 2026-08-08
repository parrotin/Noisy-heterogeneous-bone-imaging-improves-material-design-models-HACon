from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(frozen=True)
class SchedulePoint:
    step: int
    learning_rate: float
    progress: float


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        minimum_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= warmup_steps:
            raise ValueError("total steps must exceed warmup steps")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.minimum_ratio = minimum_ratio
        super().__init__(optimizer, last_epoch)

    def scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine

    def get_lr(self) -> list[float]:
        factor = self.scale(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = min(1.0, (self.last_epoch + 1) / max(1, self.warmup_steps))
        return [base_lr * factor for base_lr in self.base_lrs]


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    name: str = "adamw",
) -> Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def optimizer_learning_rate(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def schedule_trace(
    base_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> list[SchedulePoint]:
    trace: list[SchedulePoint] = []
    for step in range(total_steps):
        if step < warmup_steps:
            scale = (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        trace.append(SchedulePoint(step, base_learning_rate * scale, step / total_steps))
    return trace

