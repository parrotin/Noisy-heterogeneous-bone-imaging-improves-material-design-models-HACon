from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class AverageMeter:
    total: float = 0.0
    weight: int = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += value * weight
        self.weight += weight

    @property
    def average(self) -> float:
        return self.total / max(1, self.weight)

    def reset(self) -> None:
        self.total = 0.0
        self.weight = 0


@dataclass(frozen=True)
class StepRecord:
    epoch: int
    step: int
    global_step: int
    loss: float
    positive_similarity: float
    negative_similarity: float
    learning_rate: float
    pair_count: int
    elapsed_seconds: float


class JsonlRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: StepRecord) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True))
            stream.write("\n")


class ProgressLogger:
    def __init__(self, logger: logging.Logger, interval: int = 20) -> None:
        self.logger = logger
        self.interval = interval
        self.started = monotonic()

    def update(self, record: StepRecord) -> None:
        if record.global_step % self.interval == 0:
            self.logger.info(
                "epoch=%d step=%d global_step=%d loss=%.6f positive=%.4f negative=%.4f lr=%.8f pairs=%d elapsed=%.1f",
                record.epoch,
                record.step,
                record.global_step,
                record.loss,
                record.positive_similarity,
                record.negative_similarity,
                record.learning_rate,
                record.pair_count,
                record.elapsed_seconds,
            )


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def tensor_value(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("hacon")


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", minimum_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        self.patience = patience
        self.mode = mode
        self.minimum_delta = minimum_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = value < self.best - self.minimum_delta if self.mode == "min" else value > self.best + self.minimum_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class ScalarHistory:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, **values: float) -> None:
        for name, value in values.items():
            self.values.setdefault(name, []).append(value)

    def latest(self, name: str) -> float:
        return self.values[name][-1]

    def mean(self, name: str) -> float:
        values = self.values[name]
        return sum(values) / len(values)

    def to_mapping(self) -> dict[str, Any]:
        return {name: list(values) for name, values in self.values.items()}

