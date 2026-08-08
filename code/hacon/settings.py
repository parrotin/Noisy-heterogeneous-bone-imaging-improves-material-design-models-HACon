from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataSettings:
    manifest: str
    patch_size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    hu_window: tuple[float, float]
    metadata_missing_limit: float


@dataclass(frozen=True)
class ModelSettings:
    backbone: str
    representation_dim: int
    projection_hidden_dim: int
    projection_dim: int


@dataclass(frozen=True)
class ObjectiveSettings:
    base_temperature: float
    modulation_strength: float
    cross_site_pairs: bool
    site_modulation: bool


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    batch_size: int
    world_size: int
    gradient_accumulation: int
    learning_rate: float
    warmup_epochs: int
    weight_decay: float
    optimizer: str
    scheduler: str
    precision: str
    gradient_clip_norm: float
    workers: int
    checkpoint_every: int


@dataclass(frozen=True)
class EvaluationSettings:
    seeds: tuple[int, ...]
    ensemble_members: int
    confidence: float


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int
    data: DataSettings
    model: ModelSettings
    objective: ObjectiveSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    source: Path = field(compare=False)


def _tuple3_int(value: list[Any]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError("expected three integer values")
    return int(value[0]), int(value[1]), int(value[2])


def _tuple3_float(value: list[Any]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("expected three floating point values")
    return float(value[0]), float(value[1]), float(value[2])


def _tuple2_float(value: list[Any]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("expected two floating point values")
    return float(value[0]), float(value[1])


def load_settings(path: str | Path) -> ExperimentSettings:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    data = raw["data"]
    model = raw["model"]
    objective = raw["objective"]
    training = raw["training"]
    evaluation = raw["evaluation"]
    return ExperimentSettings(
        seed=int(raw["seed"]),
        data=DataSettings(
            manifest=str(data["manifest"]),
            patch_size=_tuple3_int(data["patch_size"]),
            spacing=_tuple3_float(data["spacing"]),
            hu_window=_tuple2_float(data["hu_window"]),
            metadata_missing_limit=float(data["metadata_missing_limit"]),
        ),
        model=ModelSettings(
            backbone=str(model["backbone"]),
            representation_dim=int(model["representation_dim"]),
            projection_hidden_dim=int(model["projection_hidden_dim"]),
            projection_dim=int(model["projection_dim"]),
        ),
        objective=ObjectiveSettings(
            base_temperature=float(objective["base_temperature"]),
            modulation_strength=float(objective["modulation_strength"]),
            cross_site_pairs=bool(objective["cross_site_pairs"]),
            site_modulation=bool(objective["site_modulation"]),
        ),
        training=TrainingSettings(
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            world_size=int(training["world_size"]),
            gradient_accumulation=int(training["gradient_accumulation"]),
            learning_rate=float(training["learning_rate"]),
            warmup_epochs=int(training["warmup_epochs"]),
            weight_decay=float(training["weight_decay"]),
            optimizer=str(training["optimizer"]),
            scheduler=str(training["scheduler"]),
            precision=str(training["precision"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            workers=int(training["workers"]),
            checkpoint_every=int(training["checkpoint_every"]),
        ),
        evaluation=EvaluationSettings(
            seeds=tuple(int(seed) for seed in evaluation["seeds"]),
            ensemble_members=int(evaluation["ensemble_members"]),
            confidence=float(evaluation["confidence"]),
        ),
        source=source,
    )

