from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hacon.data.augmentations import training_augmentations
from hacon.data.dataset import BoneVolumeDataset, collate_volume_samples
from hacon.data.records import filter_metadata, read_manifest
from hacon.data.samplers import RegionSiteBatchSampler
from hacon.models.encoder import HAConEncoder
from hacon.objectives.contrastive import HeterogeneityAwareInfoNCE
from hacon.objectives.site_profiles import SiteProfile, scanner_dissimilarity
from hacon.settings import load_settings
from hacon.training.distributed import finalize_distributed, initialize_distributed, wrap_distributed
from hacon.training.pretrain import Pretrainer, PretrainingOptions
from hacon.training.schedule import WarmupCosineScheduler, build_optimizer
from hacon.training.state import configure_logging, set_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="hacon-train")
    value.add_argument("--config", default="configs/main.yaml")
    value.add_argument("--output", default="runs/main")
    value.add_argument("--resume", default=None)
    value.add_argument("--deterministic", action="store_true")
    return value


def initial_site_profiles(sites: int) -> list[SiteProfile]:
    return [
        SiteProfile(
            site_id=index,
            hu_offset=float(index),
            noise_power_spectrum=torch.linspace(0.0, 1.0, 64) * (index + 1),
            resolution=0.5 + index * 0.1,
        )
        for index in range(sites)
    ]


def main() -> None:
    arguments = parser().parse_args()
    settings = load_settings(arguments.config)
    logger = configure_logging()
    context = initialize_distributed()
    set_seed(settings.seed + context.rank, arguments.deterministic)
    records = filter_metadata(
        read_manifest(settings.data.manifest),
        settings.data.metadata_missing_limit,
    )
    dataset = BoneVolumeDataset(
        records,
        settings.data.patch_size,
        training_augmentations(),
    )
    if context.initialized:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=settings.seed,
            drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=settings.training.batch_size // context.world_size,
            sampler=sampler,
            num_workers=settings.training.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_volume_samples,
        )
    else:
        batch_sampler = RegionSiteBatchSampler(
            records,
            settings.training.batch_size,
            seed=settings.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=settings.training.workers,
            pin_memory=True,
            collate_fn=collate_volume_samples,
        )
    model = HAConEncoder(
        representation_dim=settings.model.representation_dim,
        projection_hidden_dim=settings.model.projection_hidden_dim,
        projection_dim=settings.model.projection_dim,
    )
    wrapped = wrap_distributed(model, context)
    optimizer = build_optimizer(
        wrapped,
        settings.training.learning_rate,
        settings.training.weight_decay,
        settings.training.optimizer,
    )
    total_steps = len(loader) * settings.training.epochs
    warmup_steps = len(loader) * settings.training.warmup_epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    sites = max(record.site_index for record in records) + 1
    distance = scanner_dissimilarity(initial_site_profiles(sites))
    objective = HeterogeneityAwareInfoNCE(
        settings.objective.base_temperature,
        settings.objective.modulation_strength,
    )
    options = PretrainingOptions(
        epochs=settings.training.epochs,
        gradient_accumulation=settings.training.gradient_accumulation,
        gradient_clip_norm=settings.training.gradient_clip_norm,
        precision=settings.training.precision,
        checkpoint_every=settings.training.checkpoint_every,
        output_directory=Path(arguments.output),
        seed=settings.seed,
        cross_site_pairs=settings.objective.cross_site_pairs,
        site_modulation=settings.objective.site_modulation,
    )
    trainer = Pretrainer(
        wrapped,
        objective,
        optimizer,
        scheduler,
        distance,
        context,
        options,
        logger,
    )
    try:
        trainer.fit(loader)
    finally:
        finalize_distributed(context)


if __name__ == "__main__":
    main()
