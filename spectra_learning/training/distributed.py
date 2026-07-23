import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str = "torch"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed_from_env() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device("cuda", local_rank),
            )
        else:
            dist.init_process_group(backend="gloo")
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.is_distributed and context.backend == "torch":
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.is_distributed and context.backend == "torch":
        dist.barrier()


def any_rank(value: bool, context: DistributedContext) -> bool:
    if not context.is_distributed or context.backend != "torch":
        return value
    flag = torch.tensor(int(value), device=context.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def max_across_ranks(value: float, context: DistributedContext) -> float:
    if not context.is_distributed or context.backend != "torch":
        return value
    scalar = torch.tensor(value, device=context.device)
    dist.all_reduce(scalar, op=dist.ReduceOp.MAX)
    return float(scalar.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def wrap_distributed_model(
    model: torch.nn.Module,
    context: DistributedContext,
    *,
    static_graph: bool = True,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not context.is_distributed or context.backend != "torch":
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        output_device=context.local_rank if context.device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
        static_graph=static_graph,
    )


def reduce_metric_tensors(
    metrics: dict[str, torch.Tensor],
    context: DistributedContext,
) -> dict[str, torch.Tensor]:
    if not context.is_distributed or context.backend != "torch":
        return metrics
    reduced = {}
    for key, value in metrics.items():
        reduced_value = value.detach().clone()
        dist.all_reduce(reduced_value, op=dist.ReduceOp.AVG)
        reduced[key] = reduced_value
    return reduced
