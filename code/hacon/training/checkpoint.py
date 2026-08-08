from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(frozen=True)
class TrainingPosition:
    epoch: int
    global_step: int
    samples_seen: int
    best_metric: float


@dataclass(frozen=True)
class RestoreResult:
    position: TrainingPosition
    seed: int
    extra: dict[str, Any]


def random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    position: TrainingPosition,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "position": asdict(position),
        "seed": seed,
        "random_state": random_state(),
        "extra": {} if extra is None else extra,
    }


def atomic_save(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    position: TrainingPosition,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = checkpoint_payload(model, optimizer, scheduler, scaler, position, seed, extra)
    atomic_save(path, payload)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    map_location: str | torch.device = "cpu",
) -> RestoreResult:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload["scaler"] is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_random_state(payload["random_state"])
    position = TrainingPosition(**payload["position"])
    return RestoreResult(position, int(payload["seed"]), dict(payload["extra"]))


def latest_checkpoint(directory: str | Path) -> Path | None:
    root = Path(directory)
    candidates = sorted(root.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None

