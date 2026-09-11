"""Small single-node distributed-training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def ddp_local_mean_scale(
    local_count: torch.Tensor,
    global_count: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Scale a local mean so DDP's gradient average is a global mean."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return world_size * local_count / global_count.clamp_min(1)


@dataclass(frozen=True)
class DistributedContext:
    """Process topology and the process groups used by the trainer."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    object_group: Any = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()


def initialize_distributed(system_cfg) -> DistributedContext:
    """Initialize torch.distributed from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size == 1:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(system_cfg.device),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA for the NCCL backend.")

    device_index = local_rank
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible."
        )

    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)

    timeout = timedelta(minutes=int(system_cfg.get("distributed_timeout_minutes", 60)))
    backend = system_cfg.get("distributed_backend", "nccl")
    dist.init_process_group(backend=backend, timeout=timeout)

    # Adaptive ECE needs all confidence values.  Gather those CPU objects on a
    # Gloo group so NCCL does not stage a potentially large object through VRAM.
    object_group = (
        dist.group.WORLD
        if backend == "gloo"
        else dist.new_group(backend="gloo", timeout=timeout)
    )
    return DistributedContext(rank, local_rank, world_size, device, object_group)


def destroy_distributed() -> None:
    """Tear down the default process group without adding a final barrier."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class DistributedEvalSampler(Sampler[int]):
    """Partition an evaluation dataset exactly, without padding or duplication.

    Interleaved ranges balance ordered validation sources and sequence lengths
    across ranks. Beam-search striding is applied within each local partition.
    """

    def __init__(
        self,
        dataset,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        size = len(dataset)
        self.size = size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1
