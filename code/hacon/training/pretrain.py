from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from hacon.data.dataset import VolumeSample
from hacon.models.encoder import HAConEncoder
from hacon.objectives.contrastive import HeterogeneityAwareInfoNCE
from hacon.objectives.pair_mining import cross_site_positive_pairs, within_site_positive_pairs
from hacon.training.checkpoint import TrainingPosition, save_checkpoint
from hacon.training.distributed import DistributedContext, all_reduce_mean, unwrap_model
from hacon.training.schedule import optimizer_learning_rate
from hacon.training.state import JsonlRecorder, ProgressLogger, StepRecord, tensor_value


@dataclass(frozen=True)
class PretrainingOptions:
    epochs: int
    gradient_accumulation: int
    gradient_clip_norm: float
    precision: str
    checkpoint_every: int
    output_directory: Path
    seed: int
    cross_site_pairs: bool = True
    site_modulation: bool = True


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    mean_loss: float
    mean_positive_similarity: float
    mean_negative_similarity: float
    pairs: int
    steps: int


class Pretrainer:
    def __init__(
        self,
        model: nn.Module,
        objective: HeterogeneityAwareInfoNCE,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        distance_matrix: Tensor,
        context: DistributedContext,
        options: PretrainingOptions,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distance_matrix = distance_matrix.to(context.device)
        self.context = context
        self.options = options
        self.logger = logger
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=options.precision == "fp16" and context.device.type == "cuda"
        )
        self.recorder = JsonlRecorder(options.output_directory / "training.jsonl")
        self.progress = ProgressLogger(logger)
        self.global_step = 0
        self.samples_seen = 0
        self.best_loss = float("inf")

    def _autocast_dtype(self) -> torch.dtype:
        if self.options.precision == "bf16":
            return torch.bfloat16
        return torch.float16

    def _pairs(self, sample: VolumeSample) -> object:
        if self.options.cross_site_pairs:
            return cross_site_positive_pairs(sample.region, sample.site)
        return within_site_positive_pairs(sample.region, sample.site)

    def _forward(self, sample: VolumeSample) -> tuple[Tensor, Tensor, Tensor, int]:
        model = self.model
        with torch.autocast(
            device_type=self.context.device.type,
            dtype=self._autocast_dtype(),
            enabled=self.options.precision in {"fp16", "bf16"} and self.context.device.type == "cuda",
        ):
            output = model(sample.image)
            if not isinstance(output, tuple):
                raise TypeError("HACon model must return representation and projection")
            _, projections = output
            pairs = self._pairs(sample)
            distance = self.distance_matrix
            if not self.options.site_modulation:
                distance = torch.zeros_like(distance)
            objective_output = self.objective(projections, sample.site, pairs, distance)
            loss = objective_output.loss / self.options.gradient_accumulation
        return (
            loss,
            objective_output.positive_similarity,
            objective_output.negative_similarity,
            objective_output.pair_count,
        )

    def _backward(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.options.gradient_clip_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def train_epoch(self, loader: DataLoader[VolumeSample], epoch: int) -> EpochResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        positives: list[float] = []
        negatives: list[float] = []
        total_pairs = 0
        started = monotonic()
        for step, sample in enumerate(loader):
            sample = VolumeSample(
                image=sample.image.to(self.context.device, non_blocking=True),
                site=sample.site.to(self.context.device, non_blocking=True),
                region=sample.region.to(self.context.device, non_blocking=True),
                record_index=sample.record_index.to(self.context.device, non_blocking=True),
            )
            loss, positive, negative, pair_count = self._forward(sample)
            self._backward(loss)
            if (step + 1) % self.options.gradient_accumulation == 0:
                self._optimizer_step()
            reduced_loss = all_reduce_mean(loss * self.options.gradient_accumulation, self.context)
            reduced_positive = all_reduce_mean(positive, self.context)
            reduced_negative = all_reduce_mean(negative, self.context)
            loss_value = tensor_value(reduced_loss)
            positive_value = tensor_value(reduced_positive)
            negative_value = tensor_value(reduced_negative)
            losses.append(loss_value)
            positives.append(positive_value)
            negatives.append(negative_value)
            total_pairs += pair_count
            self.global_step += 1
            self.samples_seen += sample.image.shape[0] * self.context.world_size
            record = StepRecord(
                epoch=epoch,
                step=step,
                global_step=self.global_step,
                loss=loss_value,
                positive_similarity=positive_value,
                negative_similarity=negative_value,
                learning_rate=optimizer_learning_rate(self.optimizer),
                pair_count=pair_count,
                elapsed_seconds=monotonic() - started,
            )
            if self.context.primary:
                self.recorder.append(record)
                self.progress.update(record)
        return EpochResult(
            epoch=epoch,
            mean_loss=sum(losses) / len(losses),
            mean_positive_similarity=sum(positives) / len(positives),
            mean_negative_similarity=sum(negatives) / len(negatives),
            pairs=total_pairs,
            steps=len(losses),
        )

    def _save(self, epoch: int, result: EpochResult) -> None:
        if not self.context.primary:
            return
        self.best_loss = min(self.best_loss, result.mean_loss)
        position = TrainingPosition(
            epoch=epoch,
            global_step=self.global_step,
            samples_seen=self.samples_seen,
            best_metric=self.best_loss,
        )
        path = self.options.output_directory / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            path,
            unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            position,
            self.options.seed,
            {"epoch_result": result.__dict__},
        )

    def fit(self, loader: DataLoader[VolumeSample], start_epoch: int = 0) -> list[EpochResult]:
        history: list[EpochResult] = []
        for epoch in range(start_epoch, self.options.epochs):
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            result = self.train_epoch(loader, epoch)
            history.append(result)
            self.logger.info(
                "epoch=%d loss=%.6f positive=%.4f negative=%.4f pairs=%d",
                epoch,
                result.mean_loss,
                result.mean_positive_similarity,
                result.mean_negative_similarity,
                result.pairs,
            )
            if (epoch + 1) % self.options.checkpoint_every == 0:
                self._save(epoch + 1, result)
        return history

