from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import TypeVar

import torch
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel


ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized: bool

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(timeout_minutes: int = 30) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    initialized = world_size > 1
    if initialized and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistributedContext(rank, local_rank, world_size, device, initialized)


def finalize_distributed(context: DistributedContext) -> None:
    if context.initialized and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def wrap_distributed(
    model: ModuleT,
    context: DistributedContext,
    find_unused_parameters: bool = False,
) -> ModuleT | DistributedDataParallel:
    model = model.to(context.device)
    if not context.initialized:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        output_device=context.local_rank if context.device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=True,
    )


def all_reduce_mean(value: Tensor, context: DistributedContext) -> Tensor:
    result = value.detach().clone()
    if context.initialized:
        torch.distributed.all_reduce(result, op=ReduceOp.SUM)
        result /= context.world_size
    return result


def all_reduce_sum(value: Tensor, context: DistributedContext) -> Tensor:
    result = value.detach().clone()
    if context.initialized:
        torch.distributed.all_reduce(result, op=ReduceOp.SUM)
    return result


def all_gather_tensor(value: Tensor, context: DistributedContext) -> Tensor:
    if not context.initialized:
        return value
    gathered = [torch.empty_like(value) for _ in range(context.world_size)]
    torch.distributed.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


def broadcast_object(value: object, context: DistributedContext) -> object:
    values = [value]
    if context.initialized:
        torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def synchronize(context: DistributedContext) -> None:
    if context.initialized:
        torch.distributed.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model

